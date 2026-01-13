# Journarixv

Local Arxiv newspaper generator. Fetches papers, clusters them by topic, finds benchmarks, and generates a daily HTML report with a network graph.

## Features

- **Daily Digest**: Fetches the latest papers from Arxiv (default: `cs.*`).
- **Smart Clustering**: Uses TF-IDF and K-Means to group related papers.
- **Graph Visualization**: Generates a citation and filiation network graph.
- **Benchmark Tracking**: Automatically detects common benchmarks across papers.

## Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: You'll need to generate your own `requirements.txt` based on imports if not provided)*

## Usage

Run the main script to generate today's newspaper:

```bash
python main.py
```

### Options

- `--date YYYY-MM-DD`: Fetch papers for a specific date.
- `--category cat`: Specify Arxiv category (default `cs.*`).
- `--force-fetch`: Force re-downloading data even if it exists locally.

## Output

Generated newspapers and atlases are saved in the `output/` directory. Open the HTML files in any browser.

### Viewing the Atlas
The generated SVG atlas can be very large. To generate a lighter PNG version for local viewing:
```bash
python render_atlas.py
```
This will create `output/math_stat_atlas.png`.
