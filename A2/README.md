# CS257 NLP (W'25) Assignment 1

We'll be using Python throughout the course. Make sure you have Python 3.12 or later installed. We recommend using [uv](https://docs.astral.sh/uv/) for package management.

## 1. Install uv

If you don't have uv installed:

    curl -LsSf https://astral.sh/uv/install.sh | sh

Or on macOS with Homebrew:

    brew install uv

## 2. Create a virtual environment and install dependencies

In this directory, run:

    uv sync

## 3. Install additional packages

For spacy `en_core_web_sm` run:

    uv run python -m spacy download en_core_web_sm

## 4. Run Jupyter Lab

    uv run jupyter lab

## 5. Make sure you are using the right kernel

Go to the toolbar of your .ipynb file and click on Kernel -> Change kernel. Select the Python kernel from this virtual environment.
