import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from loguru import logger
import pandas as pd


def parse_snapshot_json_file(file_path: Path):
    if not file_path:
        raise ValueError("file_path must be provided")

    logger.info(f"Parsing singular JSON metadata from {file_path}")
    records = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                records.append(
                    {
                        "id": data.get("id"),
                        "title": data.get("title"),
                        "categories": data.get("categories"),
                        "update_date": data.get("update_date"),
                    }
                )
            except json.JSONDecodeError:
                continue

    logger.info(f"Successfully parsed {len(records)} records from JSON file.")
    return pd.DataFrame(records)


def parse_oai_xml_file(file_path: Path):
    if not file_path:
        raise ValueError("file_path must be provided")

    logger.info(f"Parsing OAI-PMH XML file from {file_path}")
    records = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Clean off XML declarations if the file has concatenated raw HTTP responses 
        # instead of a well-formed single XML root.
        content = re.sub(r"<\?xml.*?\?>", "", content)
        root = ET.fromstring(f"<root>{content}</root>")

        for elem in root.iter():
            # Strip namespaces
            if elem.tag.endswith("}record") or elem.tag == "record":
                id_txt, title_txt, cats_txt, date_txt = "", "", "", ""

                for child in elem.iter():
                    tag_name = child.tag.split("}")[-1]

                    if tag_name == "identifier" and not id_txt:
                        raw_id = child.text if child.text else ""
                        id_txt = raw_id.split(":")[-1] if ":" in raw_id else raw_id
                    elif tag_name == "id" and not id_txt:
                        raw_id = child.text if child.text else ""
                        id_txt = raw_id.split(":")[-1] if ":" in raw_id else raw_id
                    elif tag_name == "title" and not title_txt:
                        title_txt = child.text if child.text else ""
                    elif tag_name == "categories" and not cats_txt:
                        cats_txt = child.text if child.text else ""
                    elif tag_name == "datestamp" and not date_txt:
                        date_txt = child.text if child.text else ""
                    elif tag_name == "updated" and not date_txt:
                        date_txt = child.text if child.text else ""

                if id_txt and title_txt:
                    records.append(
                        {
                            "id": id_txt,
                            "title": title_txt,
                            "categories": cats_txt,
                            "update_date": date_txt,
                        }
                    )
    except Exception as e:
        logger.error(f"Failed to parse {file_path}: {e}")

    logger.info(f"Successfully parsed {len(records)} records from XML file.")
    return pd.DataFrame(records)


def clean_df(df):
    """Apply text cleaning to a DataFrame, drop missing, and return cleaned payload."""

    def clean_text(text: str) -> str:
        """Applies LaTeX stripping, whitespace removal, and lowercasing."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r"\$.*?\$", "", text)
        text = text.replace("\n", " ").replace("\t", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()

    initial_count = len(df)
    logger.info(f"Starting text cleaning for {initial_count} rows...")

    df = df.copy()
    df["title"] = df["title"].apply(clean_text)

    df = df[df["title"].str.strip() != ""]
    df = df.dropna(subset=["id", "title"])

    logger.info(f"Dropped {initial_count - len(df)} invalid/empty rows. Remaining: {len(df)}.")

    return df


def embed(df, model, pool=None) -> "pd.DataFrame":
    """Generate SentenceTransformers embeddings for titles and return processed DataFrame."""

    logger.info(f"Encoding {len(df)} titles...")
    if pool is not None:
        embeddings = model.encode(df["title"].tolist(), pool=pool, show_progress_bar=True)
    else:
        embeddings = model.encode(df["title"].tolist(), show_progress_bar=True)
    
    logger.info("Successfully generated embeddings. Assigning to dataframe...")
    # Assign the new column directly without making a full copy of the dataframe
    df["embedding"] = list(embeddings)

    return df
