from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import time

from loguru import logger
import requests

OAI_BASE_URL = "https://oaipmh.arxiv.org/oai"


def extract_latest_update(snapshot_file: Path) -> str | None:
    """Scan the Kaggle snapshot and return the latest update_date found."""
    if not snapshot_file:
        raise ValueError("snapshot_file path must be provided")
    logger.info(f"Scanning {snapshot_file} for the latest update date...")
    latest_date = ""
    try:
        with open(snapshot_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    update_date = record.get("update_date")
                    if update_date and update_date > latest_date:
                        latest_date = update_date
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.error(f"Failed to scan snapshot: {e}")
        return None

    logger.info(f"Found latest update date: {latest_date}")
    return latest_date if latest_date else None


def download_kaggle_snapshot(output_path: Path) -> Path:
    """Download the latest arXiv snapshot from Kaggle and return the path."""
    if not output_path:
        raise ValueError("output_path must be provided")
        
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if not output_path.exists():
        logger.info(f"Downloading Cornell-University/arxiv dataset to {output_dir}...")
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(
                "Cornell-University/arxiv",
                path=str(output_dir),
                unzip=True,
                quiet=False,
            )
            # Kaggle extracts to arxiv-metadata-oai-snapshot.json by default
            default_extracted_file = output_dir / "arxiv-metadata-oai-snapshot.json"
            if default_extracted_file != output_path and default_extracted_file.exists():
                default_extracted_file.rename(output_path)
            logger.info("Dataset downloaded successfully.")
        except OSError as e:
            logger.error("Kaggle credentials not found! Please configure your Kaggle credentials.")
            raise e
        except Exception as e:
            logger.error(f"Failed to download from Kaggle: {e}")
            raise e

    return output_path


def fetch_oai_updates(output_path: Path, from_date: str, to_date: str):
    """Target fetch logic: saves raw XML directly to output_path"""
    if not output_path:
        raise ValueError("output_path must be provided")

    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting incremental update from {from_date} to {to_date} using OAI-PMH")

    start_date = datetime.strptime(from_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(to_date, "%Y-%m-%d").date()

    timestamp = datetime.now().strftime("%H%M%S")
    job_dir = output_path / f"arxiv_updates_{from_date}_to_{to_date}_{timestamp}"
    job_dir.mkdir(parents=True, exist_ok=True)
    
    total_records = 0

    current_date = start_date
    while current_date <= end_date:
        current_date_str = current_date.strftime("%Y-%m-%d")
        logger.info(f"Fetching updates for date: {current_date_str}...")

        params = {
            "verb": "ListRecords",
            "metadataPrefix": "arXivRaw",
            "from": current_date_str,
            "until": current_date_str,
        }

        page = 1
        while True:
            logger.info(f"Fetching page {page} for {current_date_str}...")
            try:
                response = requests.get(OAI_BASE_URL, params=params, timeout=60)
                response.raise_for_status()
                content_str = response.text

                records = re.findall(r"<record.*?>.*?</record>", content_str, re.DOTALL)

                if not records:
                    logger.info("No records found in this page.")
                else:
                    out_file = job_dir / f"{current_date_str}_page_{page}.xml"
                    with open(out_file, "w", encoding="utf-8") as f_out:
                        f_out.write(content_str)

                    logger.info(
                        f"Saved {len(records)} records for {current_date_str} to {out_file.name}"
                    )
                    total_records += len(records)

                if "<resumptionToken" not in content_str:
                    logger.info("No more pages to fetch for this date.")
                    break

                start_idx = content_str.find(">", content_str.find("<resumptionToken")) + 1
                end_idx = content_str.find("</resumptionToken>")
                token = content_str[start_idx:end_idx].strip()

                if not token:
                    logger.info("Empty resumption token found. Moving on.")
                    break

                params = {"verb": "ListRecords", "resumptionToken": token}
                page += 1
                time.sleep(5)

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to fetch data from OAI-PMH: {e}")
                raise e

        current_date += timedelta(days=1)
        time.sleep(5)  # additional sleep between day shifts

    if total_records == 0:
        logger.info("No new updates found. Cleaning up.")
        job_dir.rmdir()

    return job_dir
