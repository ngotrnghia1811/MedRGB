"""Chunk medical textbook .txt files into retrieval-ready JSONL snippets."""
import os
import re
import json
import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter


def ends_with_ending_punctuation(s: str) -> bool:
    return s.endswith((".", "?", "!"))


def concat(title: str, content: str) -> str:
    title = title.strip()
    content = content.strip()
    return title + " " + content if ends_with_ending_punctuation(title) else title + ". " + content


if __name__ == "__main__":
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    src_dir = "corpus/textbooks/en"
    out_dir = "corpus/textbooks/chunk"
    os.makedirs(out_dir, exist_ok=True)

    fnames = sorted(os.listdir(src_dir))
    for fname in tqdm.tqdm(fnames, desc="Chunking Textbooks"):
        fpath = os.path.join(src_dir, fname)
        raw_text = open(fpath).read().strip()
        chunks = text_splitter.split_text(raw_text)
        title = fname.replace(".txt", "")
        saved = [
            json.dumps({
                "id": f"{title}_{i}",
                "title": title,
                "content": re.sub(r"\s+", " ", chunk),
                "contents": concat(title, re.sub(r"\s+", " ", chunk)),
            })
            for i, chunk in enumerate(chunks)
        ]
        with open(os.path.join(out_dir, fname.replace(".txt", ".jsonl")), "w") as f:
            f.write("\n".join(saved))
