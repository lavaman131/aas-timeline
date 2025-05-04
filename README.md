# Asian American Studies Timeline

This project is a proof-of-concept digital humanities project exploring relationships between embeddings of Asian American Studies texts in a 3D vector space. It uses text embeddings generated from scholarly articles, poems, and short stories, reduces their dimensionality using PCA, and visualizes them using Plotly.js within an Astro/React frontend.

## Features

*   3D visualization of text embeddings using Plotly.js ([`src/components/Graph3D.tsx`](src/components/Graph3D.tsx)).
*   Interactive exploration: Hover over points to see details, drag to rotate, scroll to zoom.
*   Displays detailed information about the selected text ([`src/components/NodeInfo.tsx`](src/components/NodeInfo.tsx)).
*   Filtering capabilities based on text type ([`src/components/FilterControls.tsx`](src/components/FilterControls.tsx)).
*   Data processing pipeline using Python, Transformers, and Scikit-learn ([`src/scripts/generate_data.py`](src/scripts/generate_data.py)).
*   Stats sidebar displaying correlation data between texts ([`src/components/Home.tsx`](src/components/Home.tsx), [`src/data/stats.html`](src/data/stats.html)).

## Technology Stack

*   **Frontend**: Astro, React, Tailwind CSS, Plotly.js
*   **Backend/Data Processing**: Python, PyMuPDF, Transformers, Scikit-learn, NumPy
*   **Package Management**: uv (Python), npm (Node.js)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/lavaman131/aas-timeline
    cd aas-timeline
    ```

2.  **Set up Python Environment (using uv):**
    Ensure you have Python 3.10 installed (as specified in [.python-version](.python-version)).
    ```bash
    # Install base dependencies + build dependencies
    uv sync --extra build
    # Optional: install compile dependencies to enable flash-attn support (if supported)
    uv sync --extra build --extra compile
    ```

3.  **Install Node.js Dependencies:**
    Ensure you have Node.js and npm (or pnpm/yarn) installed.
    ```bash
    npm install 
    ```

## Data Generation

The 3D visualization relies on embeddings generated from PDF documents located in the [`data/`](data/) directory. Each PDF should have a corresponding `.json` file containing metadata (title, author, year, type, description).

1.  Place your PDF files and corresponding metadata JSON files in the [`data/`](data/) directory.
2.  Run the data generation script:
    ```bash
    python src/scripts/generate_data.py --data_dir ./data --output_path ./src/data/res.json
    ```
    This script will:
    *   Read text from PDFs using PyMuPDF.
    *   Load metadata from JSON files.
    *   Generate embeddings using the `jinaai/jina-embeddings-v3` model.
    *   Extract Perform truncation of embeddings to three dimensions.
    *   Normalize the embeddings.
    *   Save the processed data, including embeddings, to [`src/data/res.json`](src/data/res.json).
3. Run the stats generation script:
    ```bash
    python src/scripts/get_stats.py --data_path ./src/data/res.json --output_path ./src/data/stats.csv
    ```
    This script will:
    *   Load the processed data (including embeddings) from the specified JSON file (`res.json`).
    *   Calculate the pairwise correlation between the 3D embeddings of all texts.
    *   For each text, identify the text with the highest correlation (most similar embedding).
    *   Save a CSV file (`stats.csv`) listing each text's title, the title of its closest match, and the correlation score.
    *   Generate an HTML table fragment (`src/data/stats.html`) based on the CSV data, styled for display in the application's stats sidebar.

## Running the Development Server

To start the Astro development server:

```bash
npm run dev
```