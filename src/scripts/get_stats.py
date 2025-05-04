import json
import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="./src/data/res.json",
        help="Path to the .json file containing the data created from src/scripts/generate_data.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./src/data",
        help="Path to the output data dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)

    # Ensure output directory exists if paths include directories
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv_path = output_dir.joinpath("stats.csv")
    output_html_path = output_dir.joinpath("stats.html")

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

    # Save the raw data as CSV
    matches.to_csv(output_csv_path, index=False)

    # Define Tailwind classes based on the table-auto example
    # Use table-auto for automatic column widths, keep basic text styling
    # Added border-collapse and border styling
    table_classes = "w-full table-auto text-sm text-center rtl:text-right text-gray-300 border-collapse border border-gray-600"
    # Keep header styling, add border and center text
    thead_classes = "text-xs text-gray-100 uppercase bg-gray-700/50"
    # Keep padding and font weight for headers, add border and center text, increase padding
    th_classes = "px-6 py-4 font-medium border border-gray-500 text-center"
    # Keep padding for data cells, add border and center text, increase padding
    td_classes = "px-6 py-4 whitespace-normal border border-gray-600 text-center"
    # Keep hover effect for body rows
    tbody_tr_hover_classes = "hover:bg-gray-800/50"  # Optional: Keep or remove hover

    # Apply styling using pandas Styler
    styled = (
        matches.style
        # Add Tailwind classes to the main <table> tag
        .set_table_attributes(f'class="{table_classes}"')
        # Add classes to <th>, <td>, and apply hover effect to <tr>
        .set_table_styles(
            [
                {"selector": "thead", "props": f"class:{thead_classes};"},
                {"selector": "th", "props": f"class:{th_classes};"},
                {"selector": "td", "props": f"class:{td_classes};"},
                # Apply hover effect to body rows (optional)
                {"selector": "tbody tr", "props": f"class:{tbody_tr_hover_classes};"},
            ],
            overwrite=False,
        )
        # Format correlation to 3 decimal places
        .format({"Correlation": "{:.3f}"})
        # Add a background gradient on the Correlation column
        # Note: This adds inline styles which override Tailwind background on this column
        .background_gradient(subset=["Correlation"], cmap="coolwarm")
    )

    # Render the styled table to an HTML string
    html_table = styled.to_html(
        index=False,
        escape=False,  # Prevent escaping of class attributes
        # table_uuid="stats-table" # Optional: Add an ID for easier CSS targeting
    )

    # Save the generated HTML table fragment
    output_html_path.write_text(html_table, encoding="utf-8")

    print(f"Stats CSV saved to: {output_csv_path}")
    print(f"Stats HTML table saved to: {output_html_path}")


if __name__ == "__main__":
    main()
