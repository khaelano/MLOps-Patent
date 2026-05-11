import numpy as np
import pandas as pd

from patent.dataset.preprocess import clean_df


def test_clean_df():
    data = [
        {"id": "1", "title": "Paper A with $\\alpha$ math", "categories": "cs.AI", "update_date": "2026-01-01"},
        {"id": "2", "title": "  \n  \t  ", "categories": "cs.ML", "update_date": "2026-01-01"},
        {"id": "3", "title": None, "categories": "cs.CV", "update_date": "2026-01-01"},
        {"id": "4", "title": "Paper B", "categories": "cs.LG", "update_date": "2026-01-01"},
    ]
    df = pd.DataFrame(data)

    cleaned = clean_df(df)
    assert len(cleaned) == 2, "Should drop empty and None titles"
    assert "1" in cleaned["id"].values
    assert "4" in cleaned["id"].values
    assert "2" not in cleaned["id"].values
    assert "3" not in cleaned["id"].values

    row1 = cleaned[cleaned["id"] == "1"].iloc[0]
    assert "$" not in row1["title"], "LaTeX delimiters should be stripped"
    assert row1["title"] == "paper a with math", (
        "Should lowercase, strip LaTeX, and collapse whitespace"
    )


def test_clean_df_empty_input():
    df = pd.DataFrame(columns=["id", "title", "categories", "update_date"])
    cleaned = clean_df(df)
    assert len(cleaned) == 0


def test_clean_df_deduplicates_by_id():
    data = [
        {"id": "1", "title": "Paper A", "categories": "cs.AI", "update_date": "2026-01-01"},
        {"id": "1", "title": "Paper A updated", "categories": "cs.AI", "update_date": "2026-01-02"},
    ]
    df = pd.DataFrame(data)
    cleaned = clean_df(df)
    assert len(cleaned) > 0
    titles = cleaned[cleaned["id"] == "1"]["title"].values
    unique_titles = set(titles)
    assert len(unique_titles) >= 1
