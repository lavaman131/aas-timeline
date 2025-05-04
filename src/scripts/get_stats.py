import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/res.json",
        help="Path to the .json file containing the data created from src/scripts/generate_data.py.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="stats.csv",
        help="Path to the output csv file (including .csv extension).",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()
    data_path = Path(args.data_path)
    output_path = Path(args.output_path)

    with open(data_path, "r") as f:
        items = json.load(f)

    truncate_dim = 1024

    embedding_col_name = "embedding" if truncate_dim != 3 else "embedding_dim3"

    embeddings = np.zeros((len(items), truncate_dim))

    for i, item in enumerate(items):
        embeddings[i] = np.array(item[embedding_col_name])

    corr = np.corrcoef(embeddings)
    corr[np.diag_indices_from(corr)] = 0

    closest = np.argmax(corr, axis=1)
    matches = []
    for i, item in enumerate(items):
        matches.append([item["title"], items[closest[i]]["title"], corr[i, closest[i]]])

    matches = pd.DataFrame(matches, columns=["Title", "Closest Title", "Correlation"])

    matches.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
