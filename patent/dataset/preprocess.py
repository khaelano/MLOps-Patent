import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import typer
from loguru import logger

app = typer.Typer(help="Data preprocessing CLI for paper novelty detection pipeline.")

def clean_text(text: str) -> str:
    """Applies LaTeX stripping, whitespace removal, and lowercasing."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\$.*?\$', '', text)
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def parse_kaggle_json(file_path: str | Path) -> pd.DataFrame:
    """Parse Kaggle arxiv-metadata-oai-snapshot JSON into a DataFrame."""
    file_path = Path(file_path)
    logger.info(f"Parsing Kaggle JSON from {file_path}")
    records = []
    
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return pd.DataFrame()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                records.append({
                    'id': data.get('id'),
                    'title': data.get('title'),
                    'categories': data.get('categories'),
                    'update_date': data.get('update_date')
                })
            except json.JSONDecodeError:
                continue
                
    return pd.DataFrame(records)

def parse_oai_xml_files(file_paths: List[str | Path]) -> pd.DataFrame:
    """Parse a list of OAI-PMH format XML files into a DataFrame."""
    logger.info(f"Parsing {len(file_paths)} OAI-PMH XML files")
    records = []
    
    for file_path in file_paths:
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            for elem in root.iter():
                if elem.tag.endswith('}record') or elem.tag == 'record':
                    
                    id_txt, title_txt, cats_txt, date_txt = "", "", "", ""
                    
                    for child in elem.iter():
                        tag_name = child.tag.split('}')[-1]
                        
                        if tag_name == 'identifier' and not id_txt:
                            raw_id = child.text if child.text else ""
                            id_txt = raw_id.split(':')[-1] if ':' in raw_id else raw_id
                        elif tag_name == 'id' and not id_txt:
                            raw_id = child.text if child.text else ""
                            id_txt = raw_id.split(':')[-1] if ':' in raw_id else raw_id
                        elif tag_name == 'title' and not title_txt:
                            title_txt = child.text if child.text else ""
                        elif tag_name == 'categories' and not cats_txt:
                            cats_txt = child.text if child.text else ""
                        elif tag_name == 'datestamp' and not date_txt:
                            date_txt = child.text if child.text else ""
                        elif tag_name == 'updated' and not date_txt:
                            date_txt = child.text if child.text else ""
                            
                    if id_txt and title_txt:
                        records.append({
                            'id': id_txt,
                            'title': title_txt,
                            'categories': cats_txt,
                            'update_date': date_txt
                        })
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            
    return pd.DataFrame(records)

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate, clean text, and filter invalid records."""
    if df.empty:
        return df

    logger.info(f"Initial row count: {len(df)}")
    
    df['update_date'] = pd.to_datetime(df['update_date'], errors='coerce')
    df = df.sort_values('update_date', ascending=False)
    df = df.drop_duplicates(subset=['id'], keep='first')
    
    logger.info(f"Row count after deduplication: {len(df)}")
    
    df['title'] = df['title'].apply(clean_text)
    
    df = df[df['title'].str.strip() != ""]
    df = df.dropna(subset=['id', 'title'])
    
    logger.info(f"Row count after text cleaning and validation: {len(df)}")
    return df

def embed_titles(titles: List[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """
    Compute sentence-level embeddings using a pre-trained transformer model.
    """
    logger.info(f"Loading SentenceTransformer model ('{model_name}')...")
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer model: {e}")
        raise

    logger.info("Encoding titles...")
    embeddings = model.encode(titles, show_progress_bar=True)
    return embeddings

@app.command()
def run(
    kaggle_json: Path = typer.Option(None, help="Path to Kaggle arxiv-metadata-oai-snapshot.json"),
    xml_dir: Path = typer.Option(None, help="Directory containing OAI-PMH XML files from updates"),
    xml_file: Path = typer.Option(None, help="Path to a single OAI-PMH XML file from updates"),
    model_name: str = typer.Option("all-MiniLM-L6-v2", help="SentenceTransformer model name to use"),
    output_path: Path = typer.Option(..., help="Path to save the processed DataFrame (e.g. data/processed/data.pkl)")
):
    """
    Execute the data preprocessing pipeline.
    """
    dfs = []
    
    if kaggle_json and kaggle_json.exists():
        df_json = parse_kaggle_json(kaggle_json)
        dfs.append(df_json)
        
    if xml_dir and xml_dir.is_dir():
        xml_files = list(xml_dir.glob("*.xml"))
        if xml_files:
            df_xml = parse_oai_xml_files(xml_files)
            dfs.append(df_xml)
            
    if xml_file and xml_file.is_file():
        df_xml = parse_oai_xml_files([xml_file])
        dfs.append(df_xml)
            
    if not dfs:
        logger.error("No data sources were successfully parsed. Exiting.")
        raise typer.Exit(1)
        
    unified_df = pd.concat(dfs, ignore_index=True)
    
    cleaned_df = process_dataframe(unified_df)
    
    if cleaned_df.empty:
        logger.error("Cleaned dataframe is empty. Exiting.")
        raise typer.Exit(1)
    
    logger.info("Generating embeddings for titles...")
    embeddings = embed_titles(cleaned_df['title'].tolist(), model_name)
    cleaned_df['embedding'] = list(embeddings)
    
    logger.info(f"Saving processed dataset to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_pickle(output_path)
    logger.info("Preprocessing pipeline completed successfully.")

if __name__ == "__main__":
    app()
