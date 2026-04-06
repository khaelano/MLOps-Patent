import numpy as np
import pandas as pd
from patent.dataset.preprocess import clean_text, process_dataframe, embed_titles

def test_clean_text():
    """Test that text cleaning handles LaTeX, whitespace, and lowercasing."""
    raw_text = "This is a \n title with \t tabs and $\\alpha = 1$ math."
    expected = "this is a title with tabs and math."
    assert clean_text(raw_text) == expected
    
    empty_text = ""
    assert clean_text(empty_text) == ""

    just_math = "$\\beta$"
    assert clean_text(just_math) == ""

def test_process_dataframe():
    """Test deduplication by date and removal of empty titles."""
    data = [
        {"id": "1", "title": "Paper A", "categories": "cs.AI", "update_date": "2026-01-01"},
        {"id": "1", "title": "Paper A updated", "categories": "cs.AI", "update_date": "2026-01-02"},
        {"id": "2", "title": "  \n  ", "categories": "cs.ML", "update_date": "2026-01-01"},
        {"id": "3", "title": "Paper B", "categories": "cs.CV", "update_date": "2026-01-01"}
    ]
    df = pd.DataFrame(data)
    
    processed_df = process_dataframe(df)
    
    assert len(processed_df) == 2, "DataFrame should contain exactly 2 valid, deduplicated rows."
    
    paper_1_title = processed_df[processed_df["id"] == "1"]["title"].iloc[0]
    assert paper_1_title == "paper a updated", "Should retain the title with the most recent update_date."
    
    assert "2" not in processed_df["id"].values, "Records with effectively empty titles should be dropped."

def test_embed_titles():
    """Test embedding function successfully outputs appropriately shaped vectors."""
    titles = ["a paper about machine learning", "another document on computer vision"]
    
    embeddings = embed_titles(titles, model_name="all-MiniLM-L6-v2")
    
    assert isinstance(embeddings, np.ndarray), "Output should be a numpy array."
    assert embeddings.shape[0] == len(titles), "Output rows should match input titles length."
    assert embeddings.shape[1] == 384, "The miniLM model should output 384-dimensional embeddings by default."
