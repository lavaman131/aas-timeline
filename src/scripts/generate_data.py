from typing import List, Dict, TypedDict, Tuple
import argparse
from pathlib import Path
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import re
from sklearn.decomposition import PCA
import numpy as np
import json
import numpy.linalg as LA


def sum_pooling(
    model_output: Tuple[torch.Tensor, torch.Tensor], attention_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return (
        torch.sum(token_embeddings * input_mask_expanded, 1),
        torch.sum(input_mask_expanded, 1),
    )


def load_model(model_name: str) -> Tuple[AutoTokenizer, AutoModel]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer, model


class TextItem(TypedDict):
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
        required=True,
        help="Path to the data directory. Must contain .txt files.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
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

    tokenizer, model = load_model("jinaai/jina-embeddings-v3")
    task = "retrieval.query"

    SENTENCE_REGEX = re.compile(
        r"(?<!\w\.\w.)(?<!\b[A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?)\s|\\n"
    )

    pca = PCA(n_components=3)
    embed_dim = 32

    items = []

    for idx, file in enumerate(tqdm(list(data_dir.glob("*.txt")))):
        numerator = torch.zeros(embed_dim)
        denominator = torch.zeros(embed_dim)
        with open(file, "r") as f:
            text = f.read()

        sentences = SENTENCE_REGEX.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        encoded_input = tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )

        task_id = model._adaptation_map[task]
        adapter_mask = torch.full((len(sentences),), task_id, dtype=torch.int32)

        model_output = model(**encoded_input, adapter_mask=adapter_mask)

        embeddings, denom = sum_pooling(model_output, encoded_input["attention_mask"])

        embeddings = embeddings.cpu()

        with open(output_path, "w") as f:
            metadata = json.load(f)

        for embedding in embeddings:
            numerator += embedding
            denominator += denom

        res = numerator / (denominator + 1e-9)

        res = pca.fit_transform(res.numpy())
        res = res / (LA.norm(res, axis=1) + 1e-9)

        item = {
            "id": str(idx),
            "title": file.stem,
            "author": metadata["author"],
            "year": metadata["year"],
            "type": metadata["type"],
            "embedding": res.tolist(),
        }
        items.append(item)

    with open(output_path, "w") as f:
        json.dump(items, f)
