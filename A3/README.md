# CS257 NLP (W'26) Assignment 3: Attention and Prompting

## Running on Google Colab

You should eventually test your implementation using Google Colab GPUs. We have provided a Colab notebook template for you to run your python and shell scripts. Refer to the notebook below for more detailed instructions.

- **A3 Colab Notebook Template**
    https://colab.research.google.com/drive/1KiXRVyLatCNXoP5ohEzCy1S0SOZLe1OK?usp=sharing

If you are unfamiliar with Google Colab (e.g., uploading files or activating GPU runtime), the following tutorials by Stanford CS224N may be helpful:

- **CS224N Colab Guide**  
    https://docs.google.com/document/d/1aNFgNKmdLCMYl8guJ2shGXa-PYCOVCtv5o65Z4GfF4A/edit


## Running Locally

It is recommended to first make sure that your code works locally before running it on Google Colab. It might be more convenient to run over the test cases and debug locally.

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
