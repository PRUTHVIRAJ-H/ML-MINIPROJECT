# Quick start

Run the entire project with one command:

```bash
bash start.sh
```

The script creates `.venv`, installs `requirements.txt`, downloads the NLTK
resources, downloads the official poetry subset and FastText model when
needed, runs the complete pipeline, and creates a single HTML report. The
first run downloads approximately 350 MB.

Open `reports/index.html` after the command completes. It contains all metrics
and graphs in one self-contained view.

## Required input files

```text
data/goodreads_reviews_dedup.json
data/goodreads_books.json
lid.176.bin
```

The script downloads these automatically. They are not stored in the
repository because they are large external files.

## What the script runs

```text
1. Create or reuse .venv
2. Install Python dependencies
3. Download NLTK resources
4. Download or check required input files
5. Run code/data_prep.py
6. Run code/features.py
7. Run code/train_model.py
```

If an input file is missing, the script prints the exact missing path and
stops before running the model.
