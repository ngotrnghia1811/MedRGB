"""Chunk the StatPearls NCBI XML dump into retrieval-ready JSONL snippets."""
import os
import json
import tqdm
import xml.etree.ElementTree as ET


def ends_with_ending_punctuation(s: str) -> bool:
    return s.endswith((".", "?", "!"))


def concat(title: str, content: str) -> str:
    title = title.strip()
    content = content.strip()
    return title + " " + content if ends_with_ending_punctuation(title) else title + ". " + content


def extract_text(element) -> str:
    text = (element.text or "").strip()
    for child in element:
        text += (" " if text else "") + extract_text(child)
        if child.tail and child.tail.strip():
            text += (" " if text else "") + child.tail.strip()
    return text.strip()


def is_subtitle(element) -> bool:
    if element.tag != "p":
        return False
    children = list(element)
    if len(children) != 1 or children[0].tag != "bold":
        return False
    return not (children[0].tail and children[0].tail.strip())


def extract(fpath: str):
    fname = os.path.splitext(os.path.basename(fpath))[0]
    tree = ET.parse(fpath)
    title = tree.getroot().find(".//title").text
    sections = tree.getroot().findall(".//sec")

    saved_text = []
    j = 0
    for sec in sections:
        sec_title = sec.find("./title").text.strip()
        prefix = f"{title} -- {sec_title}"
        last_text = None
        last_json = None
        last_node = None

        for ch in sec:
            if is_subtitle(ch):
                last_text = None
                last_json = None
                sub_title = extract_text(ch)
                prefix = " -- ".join(prefix.split(" -- ")[:2] + [sub_title])
            elif ch.tag == "p":
                curr_text = extract_text(ch)
                if len(curr_text) < 200 and last_text and len(last_text + curr_text) < 1000:
                    last_text = " ".join([last_json["content"], curr_text])
                    last_json = {**last_json, "content": last_text, "contents": concat(last_json["title"], last_text)}
                    saved_text[-1] = json.dumps(last_json)
                else:
                    last_text = curr_text
                    last_json = {
                        "id": f"{fname}_{j}",
                        "title": prefix,
                        "content": curr_text,
                        "contents": concat(prefix, curr_text),
                    }
                    saved_text.append(json.dumps(last_json))
                    j += 1
            elif ch.tag == "list":
                list_texts = [extract_text(c) for c in ch]
                combined = " ".join(list_texts)
                if last_text and len(combined + last_text) < 1000:
                    last_text = " ".join([last_json["content"]] + list_texts)
                    last_json = {**last_json, "content": last_text, "contents": concat(last_json["title"], last_text)}
                    saved_text[-1] = json.dumps(last_json)
                elif len(combined) < 1000:
                    last_text = combined
                    last_json = {
                        "id": f"{fname}_{j}",
                        "title": prefix,
                        "content": last_text,
                        "contents": concat(prefix, last_text),
                    }
                    saved_text.append(json.dumps(last_json))
                    j += 1
                else:
                    last_text = None
                    last_json = None
                    for c in list_texts:
                        saved_text.append(json.dumps({"id": f"{fname}_{j}", "title": prefix, "content": c, "contents": concat(prefix, c)}))
                        j += 1
                if last_node is not None and is_subtitle(last_node):
                    prefix = f"{title} -- {sec_title}"
            last_node = ch

    return saved_text


if __name__ == "__main__":
    nxml_dir = "corpus/statpearls/statpearls_NBK430685"
    out_dir = "corpus/statpearls/chunk"
    os.makedirs(out_dir, exist_ok=True)

    fnames = sorted(f for f in os.listdir(nxml_dir) if f.endswith(".nxml"))
    for fname in tqdm.tqdm(fnames, desc="Chunking StatPearls"):
        chunks = extract(os.path.join(nxml_dir, fname))
        if chunks:
            with open(os.path.join(out_dir, fname.replace(".nxml", ".jsonl")), "w") as f:
                f.write("\n".join(chunks))
