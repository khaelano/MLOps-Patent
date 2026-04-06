from datetime import datetime, timedelta
from pathlib import Path
import xml.etree.ElementTree as ET

from patent.dataset.data_ingestion import fetch_oai_updates

def test_fetch_oai_updates_real_api(tmp_path: Path):
    """
    Test the periodic data ingestion against the real arXiv OAI-PMH API.
    We use a very recent date to keep the payload small and the test fast.
    """
    recent_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    
    fetch_oai_updates(from_date=recent_date, output_dir=tmp_path)
    
    xml_files = list(tmp_path.glob("arxiv_updates_*.xml"))
    assert len(xml_files) == 1, "Exactly one combined XML file should be created."
    
    output_file = xml_files[0]
    
    assert output_file.stat().st_size > 0, "The output XML file should not be empty."
    
    try:
        tree = ET.parse(output_file)
        root = tree.getroot()
    except ET.ParseError as e:
        assert False, f"Output file is not valid XML: {e}"
        
    assert root.tag == "collection", "The XML root should be `<collection>`."
    
    records = list(root.findall(".//record"))
