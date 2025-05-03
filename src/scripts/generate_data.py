from typing import List, Dict, TypedDict, Tuple
import argparse
from pathlib import Path
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import re
import numpy as np
import json
import numpy.linalg as LA
import pymupdf
from torch_pca import PCA
from natsort import natsorted


class TextItem(TypedDict):
    id: str
    title: str
    author: str
    year: int
    type: str
    description: str
    embedding: List[float]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Path to the data directory. Must contain .txt files.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="res.json",
        help="Path to the output json file (including .json extension).",
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = get_args()
    data_dir = Path(args.data_dir)
    assert data_dir.is_dir(), f"Data directory {data_dir} does not exist"
    output_path = Path(args.output_path)
    assert output_path.suffix == ".json", (
        f"Output path {output_path} must have a .json extension"
    )
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = "jinaai/jina-embeddings-v3"

    device_map = "cuda:0"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, device_map=device_map
    )
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, device_map=device_map
    )
    task = "text-matching"

    SENTENCE_REGEX = re.compile(
        r"(?<!\w\.\w.)(?<!\b[A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?)\s|\\n"
    )

    pca = PCA(n_components=3)

    items = []

    files = natsorted(list(data_dir.glob("*.pdf")))

    for idx, file in enumerate(tqdm(files)):
        metadata_file = file.with_suffix(".json")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        doc = pymupdf.open(file)
        text = "".join([page.get_text() for page in doc])

        sentences = SENTENCE_REGEX.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        encoded_input = tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )  # type: ignore

        task_id = model._adaptation_map[task]  # type: ignore
        adapter_mask = torch.full(
            (len(sentences),), task_id, dtype=torch.int32, device=device_map
        )

        encoded_input = {k: v.to(device_map) for k, v in encoded_input.items()}

        model_output = model(**encoded_input, adapter_mask=adapter_mask)  # type: ignore

        attn_mask = encoded_input["attention_mask"].float()

        token_embeddings = model_output[0]

        embed_dim = token_embeddings.size(-1)
        token_embeddings = token_embeddings.view(-1, embed_dim)

        pca = PCA(n_components=3)

        embeddings = pca.fit_transform(token_embeddings.float())

        embedding = torch.sum(embeddings * attn_mask.view(-1, 1), 0) / torch.clamp(
            attn_mask.sum(), min=1e-9
        )

        embedding = F.normalize(embedding, p=2, dim=-1)
        embedding = embedding.cpu().numpy()

        item: TextItem = {
            "id": str(idx),
            "title": file.stem,
            "author": metadata["author"],
            "year": metadata["year"],
            "type": metadata["type"],
            "description": metadata["description"],
            "embedding": embedding.tolist(),
        }
        items.append(item)

    with open(output_path, "w") as f:
        json.dump(items, f)


if __name__ == "__main__":
    main()
