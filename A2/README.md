# CS257 NLP (W'26) Assignment 2: Text Classification and Word2Vec

## Running on Google Colab (Recommended)

The notebook is designed to run on Google Colab. Open `hw2_dist.ipynb` in Colab and run the setup cells which will:

1. Clone the course repository
2. Install required packages (`gensim`, `nltk`)

Make sure that you set up the .py files correctly on Colab.

## Reference: Google Colab Tutorials

If you are unfamiliar with Google Colab (e.g., uploading files or enabling GPU), the following Stanford-provided tutorials may be helpful:

- **CS224N Colab Guide**  
  https://docs.google.com/document/d/1aNFgNKmdLCMYl8guJ2shGXa-PYCOVCtv5o65Z4Gf4A/edit

- **Colab Tutorial Notebook**  
  https://colab.research.google.com/drive/1vp_6RYqMhjVItz2MRhuWnBBmz4BJ30j


## Running Locally

If you prefer to run locally:

### 1. Install uv

If you don't have uv installed:

    curl -LsSf https://astral.sh/uv/install.sh | sh

Or on macOS with Homebrew:

    brew install uv

### 2. Create a virtual environment and install dependencies

In this directory, run:

    uv sync

### 3. Run Jupyter Lab

    uv run jupyter lab

### 4. Make sure you are using the right kernel

Go to the toolbar of your .ipynb file and click on Kernel -> Change kernel. Select the Python kernel from this virtual environment.
