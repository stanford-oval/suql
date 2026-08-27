# SUQL Test Suite

This directory contains the test suite for the SUQL project.

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage
```bash
pytest --cov=src/suql --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_utils.py
```

### Run specific test
```bash
pytest tests/test_utils.py::TestNumTokensFromString::test_empty_string
```

## Test Structure

- `test_utils.py` - Tests for utility functions
- `test_faiss_embedding.py` - Tests for FAISS embedding functionality
- `conftest.py` - Shared fixtures and pytest configuration

## Adding New Tests

When adding new functionality, please add corresponding tests:

1. Create a new test file following the naming convention `test_*.py`
2. Import the module you're testing
3. Write test classes and methods
4. Use fixtures from `conftest.py` when appropriate
5. Ensure tests are isolated and don't depend on external services

## Test Coverage Goals

- Target: >70% code coverage
- Critical functions: 100% coverage
- Integration tests: Cover main workflows

## Notes

- Unit tests should not require database connections
- Integration tests may require test database setup
- Mock external services (LLM APIs, databases) when possible

