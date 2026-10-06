# Input data

Put the raw input files for this project in this directory:

```text
goodreads_reviews_dedup.json
goodreads_books.json
```

The FastText language-identification model must be placed in the repository
root:

```text
../lid.176.bin
```

See [DATA_SOURCES.md](./DATA_SOURCES.md) for the source links and
download details. The current UCSD dataset page is:

<https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html>

The raw files are intentionally not included in Git because of their size.
After adding them, run the complete project from the repository root:

```bash
bash start.sh
```
