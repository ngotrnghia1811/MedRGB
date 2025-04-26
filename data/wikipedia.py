"""Chunk the HuggingFace Wikipedia dump into retrieval-ready JSONL snippets."""
import os
import re
import json
import tqdm
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter


def ends_with_ending_punctuation(s: str) -> bool:
    return s.endswith((".", "?", "!"))


def concat(title: str, content: str) -> str:
    title = title.strip()
    content = content.strip()
    return title + " " + content if ends_with_ending_punctuation(title) else title + ". " + content


if __name__ == "__main__":
    dat = load_dataset("wikipedia", "20220301.en", cache_dir="./corpus/wikipedia", trust_remote_code=True)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    out_dir = "corpus/wikipedia/chunk"
    os.makedirs(out_dir, exist_ok=True)

    batch_size = 10000
    n_batches = len(dat["train"]) // batch_size + 1
    len_just = len(str(n_batches))

    buffer = []
    for i, article in enumerate(tqdm.tqdm(dat["train"], desc="Chunking Wikipedia")):
        batch_id = i // batch_size
        out_path = os.path.join(out_dir, f"wiki20220301en{str(batch_id).rjust(len_just, '0')}.jsonl")
        if os.path.exists(out_path):
            continue
        chunks = text_splitter.split_text(article["text"].strip())
        for j, chunk in enumerate(chunks):
            buffer.append(json.dumps({
                "id": f"{article['id']}_{j}",
                "title": article["title"],
                "content": re.sub(r"\s+", " ", chunk),
                "contents": concat(article["title"], re.sub(r"\s+", " ", chunk)),
            }))
        if (i + 1) % batch_size == 0:
            with open(out_path, "w") as f:
                f.write("\n".join(buffer))
            buffer = []

    if buffer:
        out_path = os.path.join(out_dir, f"wiki20220301en{str(batch_id).rjust(len_just, '0')}.jsonl")
        with open(out_path, "w") as f:
            f.write("\n".join(buffer))
