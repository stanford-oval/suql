"""
Pytest configuration and shared fixtures
"""

import pytest
import os
from typing import Dict, Any


@pytest.fixture
def mock_db_config() -> Dict[str, Any]:
    """Fixture providing mock database configuration"""
    return {
        "db_name": "test_db",
        "user": "test_user",
        "password": "test_password",
        "host": "localhost",
        "port": 5432,
    }


@pytest.fixture
def sample_table_mapping() -> Dict[str, str]:
    """Fixture providing sample table to ID column mapping"""
    return {
        "restaurants": "_id",
        "courses": "course_id",
    }


@pytest.fixture
def sample_suql_queries() -> Dict[str, str]:
    """Fixture providing sample SUQL queries for testing"""
    return {
        "simple_select": "SELECT * FROM restaurants LIMIT 5;",
        "with_where": "SELECT * FROM restaurants WHERE rating >= 4.0 LIMIT 10;",
        "with_answer": "SELECT * FROM restaurants WHERE answer(reviews, 'is this family-friendly?') = 'Yes' LIMIT 5;",
        "with_summary": "SELECT *, summary(reviews) FROM restaurants WHERE location = 'Palo Alto' LIMIT 3;",
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables"""
    # Mock environment variables that might be needed
    monkeypatch.setenv("TESTING", "true")
    # Don't require actual API keys for unit tests
    if "OPENAI_API_KEY" not in os.environ:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

