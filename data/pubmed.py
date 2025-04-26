"""Chunk the PubMed XML.gz baseline files into retrieval-ready JSONL snippets."""
import os
import gzip
import json
import tqdm


def ends_with_ending_punctuation(s: str) -> bool:
    return s.endswith((".", "?", "!"))


def concat(title: str, content: str) -> str:
    title = title.strip()
    content = content.strip()
    return title + " " + content if ends_with_ending_punctuation(title) else title + ". " + content


def extract(gz_fpath: str):
    titles, abstracts, ids = [], [], []
    title = abs_text = pmid = ""

    for line in gzip.open(gz_fpath, "rt").read().split("\n"):
        line = line.strip()
        if line == "<Article>" or line.startswith("<Article "):
            title = abs_text = ""
        elif line == "</Article>":
            if abs_text.strip():
                titles.append(title)
                abstracts.append(abs_text)
                ids.append(pmid)
        elif line.startswith("<PMID"):
            pmid = line.strip("</>PMID ").split(">")[-1]
        elif line.startswith("<ArticleTitle>"):
            title = line[14:-15]
        elif line.startswith("<AbstractText"):
            segment = "".join(line[13:-15].split(">")[1:])
            abs_text = abs_text + " " + segment if abs_text else segment

    return titles, abstracts, ids


if __name__ == "__main__":
    baseline_dir = "corpus/pubmed/baseline"
    out_dir = "corpus/pubmed/chunk"
    os.makedirs(out_dir, exist_ok=True)

    fnames = sorted(f for f in os.listdir(baseline_dir) if f.endswith(".xml.gz"))
    for fname in tqdm.tqdm(fnames, desc="Chunking PubMed"):
        out_path = os.path.join(out_dir, fname.replace(".xml.gz", ".jsonl"))
        if os.path.exists(out_path):
            continue
        titles, abstracts, ids = extract(os.path.join(baseline_dir, fname))
        chunks = [
            json.dumps({
                "id": f"PMID:{ids[i]}",
                "title": titles[i],
                "content": abstracts[i],
                "contents": concat(titles[i], abstracts[i]),
            })
            for i in range(len(titles))
        ]
        with open(out_path, "w") as f:
            f.write("\n".join(chunks))
