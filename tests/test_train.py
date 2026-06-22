import os
import pytest
import pandas as pd
from src.generate_data import generate_ticket_data

@pytest.fixture
def setup_dummy_data():
    """Generates dummy data specifically for testing."""
    test_path = "data/test_tickets.csv"
    generate_ticket_data(num_samples=50, output_path=test_path)
    yield test_path
    if os.path.exists(test_path):
        os.remove(test_path)

def test_data_generation_format(setup_dummy_data):
    """Tests if the data loading pipeline will receive the correct schema."""
    df = pd.read_csv(setup_dummy_data)
    
    assert "text" in df.columns, "Missing 'text' column"
    assert "category" in df.columns, "Missing 'category' column"
    assert len(df) == 50, "Incorrect number of rows generated"
    
    valid_categories = {"bug", "billing", "feature request"}
    assert set(df["category"].unique()).issubset(valid_categories), "Unexpected categories found"