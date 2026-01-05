# Assignment 1: Exploring Zipf's Law

## Environment Setup

This assignment uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Install uv (if not already installed)

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative (using pip):**
```bash
pip install uv
```

### Set up the environment

```bash
cd assignments/hw1
uv sync
```

1. Open the `hw1` directory
2. Install the Python and Jupyter extensions
3. Open `hw1_dist.ipynb`
4. Select the kernel: click the kernel selector in the top-right and choose `.venv/bin/python`

## Dependencies

See `pyproject.toml` for the full list. Key packages:
- `numpy`, `matplotlib`, `pandas` for data analysis
- `powerlaw` for power law fitting and testing
- `nltk` for language processing utilities
- `datasets` for loading Hugging Face datasets
