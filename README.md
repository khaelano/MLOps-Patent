# MLOps-Patent

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A paper novelty assessor based on anomaly detection

## Background
Research novelty is one of the important aspects in the development of science; this characteristic serves as a driving force for scientific development by investigating problems that have never been researched before (Zhao & Zhang, 2025). Therefore, evaluating novelty in research is crucial.

One method to assess the novelty of a study is by using keywords. This method works by identifying keyword combinations that are uncommon within an article. If an article contains many keyword combinations that are unusual in its field, then it can be said that the article has a high novelty value (Zhao & Zhang, 2025). However, this method only considers keywords, thereby ignoring other important aspects such as semantics and citation relationships between articles.

Therefore, an anomaly detection-based novelty assessment method has emerged to evaluate the novelty of an article solely through its title. This method works by detecting anomalies (outliers) in a semantic space created using vector representations of existing research titles, where outliers are considered indicative of novelty (Jeon et al., 2023). Thus, the model maps relationships between articles based on the semantic similarity of titles. Through this approach, the model is expected to assess the novelty of a research title reliably and accurately.

## Usage

### Prerequisites

- Python 3.10
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/khaelano/MLOps-Patent.git
   cd MLOps-Patent
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   make create_environment
   source .venv/bin/activate
   make requirements
   ```

### Pipeline

The project follows a sequential data science pipeline. Each step is a Typer CLI app that can be run independently:

1. **Process raw data**
   ```bash
   python patent/dataset.py --input-path data/raw/dataset.csv --output-path data/processed/dataset.csv
   ```

2. **Generate features**
   ```bash
   python patent/features.py --input-path data/processed/dataset.csv --output-path data/processed/features.csv
   ```

3. **Train model**
   ```bash
   python patent/modeling/train.py --features-path data/processed/features.csv --labels-path data/processed/labels.csv --model-path models/model.pkl
   ```

4. **Run predictions**
   ```bash
   python patent/modeling/predict.py --features-path data/processed/test_features.csv --model-path models/model.pkl --predictions-path data/processed/test_predictions.csv
   ```

5. **Generate plots**
   ```bash
   python patent/plots.py --input-path data/processed/dataset.csv --output-path reports/figures/plot.png
   ```

 Use `--help` on any command to see available options.

### Make Commands

| Command                  | Description                                |
| ------------------------ | ------------------------------------------ |
| `make requirements`      | Install/sync Python dependencies with uv   |
| `make data`              | Run the dataset processing pipeline        |
| `make test`              | Run tests with pytest                      |
| `make lint`              | Check code style with ruff                 |
| `make format`            | Auto-format source code with ruff          |
| `make clean`             | Delete compiled Python files and caches    |
| `make create_environment`| Create a new uv virtual environment        |

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         patent and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── patent   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes patent a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## References
Zhao, Y., & Zhang, C. (2025). A review on the novelty measurements of  academic papers. https://doi.org/10.48550/arXiv.2501.17456 