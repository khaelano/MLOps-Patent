import os
import re
import time
from datetime import datetime
from pathlib import Path

import requests
import typer
from loguru import logger

app = typer.Typer(help="Data ingestion CLI for paper novelty detection pipeline (arXiv data).")

RAW_DATA_DIR = Path("data/raw")
OAI_BASE_URL = "https://oaipmh.arxiv.org/oai"


def download_kaggle_snapshot(output_dir: Path) -> None:
    """
    Core logic to bootstrap the dataset by downloading the latest arXiv snapshot from Kaggle.
    Requires Kaggle API credentials via environment variables or ~/.kaggle/kaggle.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting initial fetch of arXiv dataset to {output_dir}")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        logger.info("Authenticating with Kaggle API...")
        logger.info("Downloading Cornell-University/arxiv dataset...")

        api.dataset_download_files(
            "Cornell-University/arxiv",
            path=str(output_dir),
            unzip=True,
            quiet=False,
        )
        logger.info(f"Dataset downloaded successfully to {output_dir}")

    except OSError as e:
        logger.error(
            "Kaggle credentials not found! "
            "Please configure your Kaggle credentials (via kaggle.json or env vars)."
        )
        logger.debug(f"Details: {e}")
        raise e
    except Exception as e:
        logger.error(f"An error occurred during initial fetch: {e}")
        raise e


@app.command("initial-fetch")
def initial_fetch(
    output_dir: Path = typer.Option(
        RAW_DATA_DIR, "--output-dir", "-o", help="Directory to save the raw dataset"
    ),
):
    """
    Bootstrap the dataset by downloading the latest arXiv snapshot from Kaggle.
    """
    try:
        download_kaggle_snapshot(output_dir)
    except Exception:
        raise typer.Exit(code=1)


def fetch_oai_updates(from_date: str, output_dir: Path, to_date: str = None) -> None:
    """
    Core logic to fetch incremental updates from arXiv using the OAI-PMH interface.
    Data is merged and saved into a single XML file covering the date range.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    end_date_str = to_date if to_date else datetime.now().strftime("%Y-%m-%d")
    timestamp_str = datetime.now().strftime("%H%M%S")
    
    output_file = output_dir / f"arxiv_updates_{from_date}_to_{end_date_str}_{timestamp_str}.xml"

    logger.info(f"Starting incremental update from {from_date} to {end_date_str} using OAI-PMH")

    params = {
        "verb": "ListRecords",
        "metadataPrefix": "arXiv",
        "from": from_date,
    }
    if to_date:
        params["until"] = to_date

    page = 1
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<collection>\n')
        
        while True:
            logger.info(f"Fetching page {page}...")

            try:
                response = requests.get(OAI_BASE_URL, params=params, timeout=30)
                response.raise_for_status()

                content_str = response.text
                records = re.findall(r'<record.*?>.*?</record>', content_str, re.DOTALL)
                for r in records:
                    f.write(r + '\n')
                    
                logger.info(f"Appended {len(records)} records to {output_file}")

                if "<resumptionToken" not in content_str:
                    logger.info("No more pages to fetch. Update complete.")
                    break

                start_idx = content_str.find(">", content_str.find("<resumptionToken")) + 1
                end_idx = content_str.find("</resumptionToken>")
                token = content_str[start_idx:end_idx].strip()

                if not token:
                    logger.info("Empty resumption token found. Update complete.")
                    break

                params = {
                    "verb": "ListRecords",
                    "resumptionToken": token
                }
                page += 1

                time.sleep(5)

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to fetch data from OAI-PMH: {e}")
                raise e
                
        f.write('</collection>\n')
        
    logger.info(f"Finished saving all records to {output_file}")


@app.command("update")
def update(
    from_date: str = typer.Option(
        ...,
        "--from-date",
        "-f",
        help="Date from which to fetch updates (format: YYYY-MM-DD)",
    ),
    to_date: str = typer.Option(
        None,
        "--to-date",
        "-t",
        help="End date for the updates to fetch (format: YYYY-MM-DD). Defaults to today.",
    ),
    output_dir: Path = typer.Option(
        RAW_DATA_DIR, "--output-dir", "-o", help="Directory where XML responses will be saved"
    ),
):
    """
    Fetch incremental updates from arXiv using the OAI-PMH interface.
    Data is merged and saved into a single XML file covering the date range.
    """
    try:
        fetch_oai_updates(from_date, output_dir, to_date)
    except Exception:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
