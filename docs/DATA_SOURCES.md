# Data sources

The project uses the publicly documented Goodreads data from the UCSD Book
Graph project.

## Source repository

The original project code is available at:

<https://github.com/bridgetdaly/goodreads_ML>

## Dataset source

The current UCSD Book Graph dataset page is:

<https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html>

The older project page redirects to the current page above. The reviews and
books files used by `data_prep.py` are obtained from the Goodreads data
released through the UCSD Book Graph project.

## Automatic demonstration subset

The one-command startup uses the official poetry subset so the demonstration
does not require the multi-gigabyte complete Goodreads archives. It downloads:

- `goodreads_books_poetry.json.gz`
- `goodreads_reviews_poetry.json.gz`

The compressed files are extracted to the filenames expected by the
preprocessing script.

The verified run produced 9,663 usable reviews. Results can change if the
source data or preprocessing code changes.

## Required local files

Place the following files in these exact locations:

```text
data/goodreads_reviews_dedup.json
data/goodreads_books.json
lid.176.bin
```

`lid.176.bin` is the FastText language-identification model used to retain
English reviews. It can be obtained from the official FastText language
identification resources:

<https://fasttext.cc/docs/en/language-identification.html>

The direct download for the required binary model is:

<https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin>

These files are not committed to this repository because the raw datasets and
the language model are large external files. The startup script downloads the
poetry subset and language model automatically when they are absent.
