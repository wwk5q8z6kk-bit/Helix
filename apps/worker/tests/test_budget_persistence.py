import os
import json
import pytest
import tempfile
from core.middleware.budget_tracker import BudgetTracker

def test_budget_persistence_lifecycle():
    # Use a temporary file for storage
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        storage_path = tmp.name
    
    try:
        # 1. Initialize tracker and record some usage
        tracker = BudgetTracker(daily_budget=10.0, storage_path=storage_path)
        tracker.record_usage("gpt-4o", input_tokens=1000, output_tokens=500)
        
        expected_cost = tracker.estimate_cost("gpt-4o", 1000, 500)
        assert tracker._daily_spend["gpt-4o"] == expected_cost
        
        # 2. Verify file was created and contains data
        assert os.path.exists(storage_path)
        with open(storage_path, 'r') as f:
            data = json.load(f)
            assert data["spend"]["gpt-4o"] == expected_cost
            
        # 3. Create a NEW tracker instance pointing to same file
        tracker2 = BudgetTracker(daily_budget=10.0, storage_path=storage_path)
        
        # 4. Verify data was loaded
        assert "gpt-4o" in tracker2._daily_spend
        assert tracker2._daily_spend["gpt-4o"] == expected_cost
        
        # 5. Record more usage on second tracker
        tracker2.record_usage("gpt-4o", input_tokens=1000, output_tokens=500)
        assert tracker2._daily_spend["gpt-4o"] == expected_cost * 2
        
        # 6. Verify file was updated
        with open(storage_path, 'r') as f:
            data = json.load(f)
            assert data["spend"]["gpt-4o"] == expected_cost * 2
            
    finally:
        if os.path.exists(storage_path):
            os.remove(storage_path)

def test_budget_persistence_date_reset():
    # Verify that if the file has data from yesterday, it's ignored
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        storage_path = tmp.name
    
    try:
        # Create a "yesterday" file
        yesterday_data = {
            "date": "2020-01-01",
            "spend": {"gpt-4o": 5.0}
        }
        with open(storage_path, 'w') as f:
            json.dump(yesterday_data, f)
            
        tracker = BudgetTracker(daily_budget=10.0, storage_path=storage_path)
        
        # Should be empty because date mismatch
        assert tracker._daily_spend == {}
        
    finally:
        if os.path.exists(storage_path):
            os.remove(storage_path)
