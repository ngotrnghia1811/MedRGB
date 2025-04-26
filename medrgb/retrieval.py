import os
import json
import torch
import tqdm
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling

from medrgb.config import CORPUS_NAMES, RETRIEVER_NAMES


class CLSSentenceTransformer(SentenceTransformer):
    """SentenceTransformer variant that uses CLS pooling instead of mean pooling.
    Required for MedCPT and SPECTER which expect CLS representations."""

    def _load_auto_model(self, model_name_or_path, *args, **kwargs):
        print(f"No sentence-transformers model found with name {model_name_or_path}. Creating with CLS pooling.")
        transformer_model = Transformer(model_name_or_path)
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), "cls")
        return [transformer_model, pooling_model]


def embed_corpus(chunk_dir: str, index_dir: str, model_name: str, **kwargs) -> int:
    """Encode all chunk JSONL files with the given retriever model and save as .npy arrays.

    Returns the embedding dimension.
    """
    save_dir = os.path.join(index_dir, "embedding")
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "contriever" in model_name.lower():
        model = SentenceTransformer(model_name, device=device)
    else:
        model = CLSSentenceTransformer(model_name, device=device)
    model.eval()

    fnames = sorted([f for f in os.listdir(chunk_dir) if f.endswith(".jsonl")])

    with torch.no_grad():
        for fname in tqdm.tqdm(fnames, desc=f"Embedding {model_name}"):
            fpath = os.path.join(chunk_dir, fname)
            save_path = os.path.join(save_dir, fname.replace(".jsonl", ".npy"))
            if os.path.exists(save_path):
                continue
            raw = open(fpath).read().strip()
            if not raw:
                continue
            items = [json.loads(line) for line in raw.split("\n")]
            if "specter" in model_name.lower():
                texts = [model.tokenizer.sep_token.join([it["title"], it["content"]]) for it in items]
            elif "contriever" in model_name.lower():
                texts = [". ".join([it["title"], it["content"]]).replace("..", ".").replace("?.", "?") for it in items]
            elif "medcpt" in model_name.lower():
                texts = [[it["title"], it["content"]] for it in items]
            else:
                texts = [it["content"] for it in items]
            embeddings = model.encode(texts, **kwargs)
            np.save(save_path, embeddings)

        dim = model.encode([""], **kwargs).shape[-1]

    return dim


def build_faiss_index(index_dir: str, model_name: str, h_dim: int = 768) -> faiss.Index:
    """Build a FAISS index from the pre-computed .npy embeddings."""
    if "specter" in model_name.lower():
        index = faiss.IndexFlatL2(h_dim)
    else:
        index = faiss.IndexFlatIP(h_dim)

    meta_path = os.path.join(index_dir, "metadatas.jsonl")
    with open(meta_path, "w") as f:
        f.write("")

    emb_dir = os.path.join(index_dir, "embedding")
    for fname in tqdm.tqdm(sorted(os.listdir(emb_dir)), desc="Building FAISS index"):
        embeddings = np.load(os.path.join(emb_dir, fname))
        index.add(embeddings)
        source = fname.replace(".npy", "")
        with open(meta_path, "a") as f:
            f.write("\n".join(json.dumps({"index": i, "source": source}) for i in range(len(embeddings))) + "\n")

    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    return index


class Retriever:
    """Single-retriever, single-corpus dense or BM25 retriever."""

    def __init__(self, retriever_name: str, corpus_name: str, db_dir: str = "./corpus"):
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir

        os.makedirs(db_dir, exist_ok=True)
        self.chunk_dir = os.path.join(db_dir, corpus_name, "chunk")

        if not os.path.exists(self.chunk_dir):
            print(f"Cloning {corpus_name} corpus from HuggingFace...")
            os.system(f"git clone https://huggingface.co/datasets/MedRAG/{corpus_name} {os.path.join(db_dir, corpus_name)}")
            if corpus_name == "statpearls":
                print("Downloading statpearls NCBI dump...")
                os.system(f"wget https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz -P {os.path.join(db_dir, corpus_name)}")
                os.system(f"tar -xzvf {os.path.join(db_dir, corpus_name, 'statpearls_NBK430685.tar.gz')} -C {os.path.join(db_dir, corpus_name)}")
                print("Chunking statpearls corpus...")
                os.system("python data/statpearls.py")

        article_encoder = retriever_name.replace("Query-Encoder", "Article-Encoder")
        self.index_dir = os.path.join(db_dir, corpus_name, "index", article_encoder)

        if "bm25" in retriever_name.lower():
            self._init_bm25()
        else:
            self._init_dense(article_encoder)

    def _init_bm25(self):
        from pyserini.search.lucene import LuceneSearcher

        self.metadatas = None
        self.embedding_function = None
        if os.path.exists(self.index_dir):
            self.index = LuceneSearcher(self.index_dir)
        else:
            os.system(
                f"python -m pyserini.index.lucene --collection JsonCollection "
                f"--input {self.chunk_dir} --index {self.index_dir} "
                f"--generator DefaultLuceneDocumentGenerator --threads 16"
            )
            self.index = LuceneSearcher(self.index_dir)

    def _init_dense(self, article_encoder: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if os.path.exists(os.path.join(self.index_dir, "faiss.index")):
            self.index = faiss.read_index(os.path.join(self.index_dir, "faiss.index"))
            self.metadatas = [
                json.loads(line)
                for line in open(os.path.join(self.index_dir, "metadatas.jsonl")).read().strip().split("\n")
            ]
        else:
            print(f"Embedding {self.corpus_name} with {article_encoder}...")
            h_dim = embed_corpus(self.chunk_dir, self.index_dir, article_encoder)
            print(f"Building FAISS index (dim={h_dim})...")
            self.index = build_faiss_index(self.index_dir, article_encoder, h_dim)
            self.metadatas = [
                json.loads(line)
                for line in open(os.path.join(self.index_dir, "metadatas.jsonl")).read().strip().split("\n")
            ]

        if "contriever" in self.retriever_name.lower():
            self.embedding_function = SentenceTransformer(self.retriever_name, device=device)
        else:
            self.embedding_function = CLSSentenceTransformer(self.retriever_name, device=device)
        self.embedding_function.eval()

    def get_relevant_documents(self, question: str, k: int = 32, **kwargs):
        assert isinstance(question, str)

        if "bm25" in self.retriever_name.lower():
            hits = self.index.search(question, k=k)
            scores = np.array([h.score for h in hits])
            indices = [
                {"source": "_".join(h.docid.split("_")[:-1]), "index": int(h.docid.split("_")[-1])}
                for h in hits
            ]
        else:
            with torch.no_grad():
                query_embed = self.embedding_function.encode([question], **kwargs)
            distances, ann_indices = self.index.search(query_embed, k)
            indices = [self.metadatas[i] for i in ann_indices[0]]
            scores = distances[0]

        texts = self._indices_to_text(indices)
        return texts, scores.tolist()

    def _indices_to_text(self, indices):
        return [
            json.loads(open(os.path.join(self.chunk_dir, idx["source"] + ".jsonl")).read().strip().split("\n")[idx["index"]])
            for idx in indices
        ]


class RetrievalSystem:
    """Multi-retriever, multi-corpus retrieval system with optional RRF fusion."""

    def __init__(self, retriever_name: str = "MedCPT", corpus_name: str = "Textbooks", db_dir: str = "./corpus"):
        assert corpus_name in CORPUS_NAMES, f"Unknown corpus: {corpus_name}"
        assert retriever_name in RETRIEVER_NAMES, f"Unknown retriever: {retriever_name}"

        self.retriever_name = retriever_name
        self.corpus_name = corpus_name

        self.retrievers = [
            [Retriever(ret, corp, db_dir) for corp in CORPUS_NAMES[corpus_name]]
            for ret in RETRIEVER_NAMES[retriever_name]
        ]

    def retrieve(self, question: str, k: int = 32, rrf_k: int = 100):
        """Retrieve top-k documents for a question."""
        k_ = max(k * 2, 100) if "RRF" in self.retriever_name else k

        texts_all, scores_all = [], []
        for ret_group in self.retrievers:
            texts_group, scores_group = [], []
            for retriever in ret_group:
                t, s = retriever.get_relevant_documents(question, k=k_)
                texts_group.extend(t)
                scores_group.extend(s)
            texts_all.append(texts_group)
            scores_all.append(scores_group)

        return self._merge(texts_all, scores_all, k=k, rrf_k=rrf_k)

    def _merge(self, texts_all, scores_all, k: int = 32, rrf_k: int = 100):
        """Merge results using RRF or direct score ordering."""
        rrf_dict = {}

        for i, (texts, scores) in enumerate(zip(texts_all, scores_all)):
            retriever_name = RETRIEVER_NAMES[self.retriever_name][i]
            if "specter" in retriever_name.lower():
                sorted_idx = np.argsort(scores)
            else:
                sorted_idx = np.argsort(scores)[::-1]

            sorted_texts = [texts[j] for j in sorted_idx]
            for rank, item in enumerate(sorted_texts):
                doc_id = item["id"]
                rrf_score = 1 / (rrf_k + rank + 1)
                if doc_id in rrf_dict:
                    rrf_dict[doc_id]["score"] += rrf_score
                    rrf_dict[doc_id]["count"] += 1
                else:
                    rrf_dict[doc_id] = {
                        "id": doc_id,
                        "title": item["title"],
                        "content": item["content"],
                        "score": rrf_score,
                        "count": 1,
                    }

        sorted_results = sorted(rrf_dict.values(), key=lambda x: x["score"], reverse=True)

        if len(texts_all) == 1:
            sorted_idx = np.argsort(scores_all[0])[::-1]
            top_texts = [texts_all[0][j] for j in sorted_idx[:k]]
            top_scores = [scores_all[0][j] for j in sorted_idx[:k]]
            return top_texts, top_scores

        top_texts = [{"id": r["id"], "title": r["title"], "content": r["content"]} for r in sorted_results[:k]]
        top_scores = [r["score"] for r in sorted_results[:k]]
        return top_texts, top_scores
