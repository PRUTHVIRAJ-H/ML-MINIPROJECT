#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"

step() {
    printf "\n\033[1;34m==> %s\033[0m\n" "$1"
}

fail() {
    printf "\n\033[1;31mERROR: %s\033[0m\n" "$1" >&2
    exit 1
}

trap 'fail "The startup script stopped at line ${LINENO}."' ERR

cd "$PROJECT_DIR"

step "Checking Python"
command -v python3 >/dev/null 2>&1 || fail "Python 3 is required."
python3 --version

step "Creating virtual environment"
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

step "Installing Python requirements"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

step "Downloading NLTK resources"
python - <<'PY'
import nltk

resources = [
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "vader_lexicon",
    "stopwords",
    "wordnet",
]

for resource in resources:
    nltk.download(resource, quiet=True)
print(f"Ready: {len(resources)} NLTK resources")
PY

step "Checking input files"
mkdir -p data
download_file() {
    local url="$1"
    local output="$2"
    local temporary="${output}.download"
    printf "Downloading %s...\n" "$output"
    curl --fail --location --retry 3 --progress-bar "$url" -o "$temporary"
    mv "$temporary" "$output"
}

if ! command -v curl >/dev/null 2>&1; then
    fail "curl is required to download the official input data."
fi

if [[ ! -f data/goodreads_books.json ]]; then
    download_file \
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_books_poetry.json.gz" \
        "/tmp/goodreads_books_poetry.json.gz"
    gzip -dc /tmp/goodreads_books_poetry.json.gz > data/goodreads_books.json
    rm -f /tmp/goodreads_books_poetry.json.gz
fi

if [[ ! -f data/goodreads_reviews_dedup.json ]]; then
    download_file \
        "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz" \
        "/tmp/goodreads_reviews_poetry.json.gz"
    gzip -dc /tmp/goodreads_reviews_poetry.json.gz > data/goodreads_reviews_dedup.json
    rm -f /tmp/goodreads_reviews_poetry.json.gz
fi

if [[ ! -f lid.176.bin ]]; then
    download_file \
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin" \
        "lid.176.bin"
fi

printf "All required input files are ready.\n"

step "Preparing raw data"
python code/data_prep.py

step "Creating features"
python code/features.py

step "Training and evaluating the model"
python code/train_model.py

printf "\n\033[1;32mProject completed successfully.\033[0m\n"
printf "Open the complete results here: %s/reports/index.html\n" "$PROJECT_DIR"
