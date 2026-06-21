"""
Unit tests for utility functions in suql.utils
"""

import pytest
from suql.utils import (
    num_tokens_from_string,
    chunk_text,
    compute_sha256,
)


class TestNumTokensFromString:
    """Test cases for num_tokens_from_string function"""

    def test_empty_string(self):
        """Test that empty string returns 0 tokens"""
        result = num_tokens_from_string("")
        assert result == 0

    def test_simple_string(self):
        """Test token count for a simple string"""
        text = "Hello world"
        result = num_tokens_from_string(text)
        assert result > 0
        assert isinstance(result, int)

    def test_long_string(self):
        """Test token count for a longer string"""
        text = "This is a longer string with multiple words and sentences. " * 10
        result = num_tokens_from_string(text)
        assert result > 0


class TestChunkText:
    """Test cases for chunk_text function"""

    def test_empty_text(self):
        """Test chunking empty text"""
        result = chunk_text("", k=10)
        assert result == [""]

    def test_simple_chunking(self):
        """Test basic text chunking"""
        text = "This is a test string with multiple words"
        result = chunk_text(text, k=5, use_spacy=False)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_chunking_with_spacy(self):
        """Test chunking with spacy (if available)"""
        text = "This is a test. This is another sentence."
        result = chunk_text(text, k=5, use_spacy=True)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_zero_chunk_size(self):
        """Test that k=0 returns original text"""
        text = "Test string"
        result = chunk_text(text, k=0)
        assert result == [text]


class TestComputeSha256:
    """Test cases for compute_sha256 function"""

    def test_basic_hash(self):
        """Test SHA256 hash computation"""
        text = "test string"
        result = compute_sha256(text)
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 produces 64 character hex string

    def test_empty_string_hash(self):
        """Test hash of empty string"""
        result = compute_sha256("")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_consistency(self):
        """Test that same input produces same hash"""
        text = "consistent test"
        hash1 = compute_sha256(text)
        hash2 = compute_sha256(text)
        assert hash1 == hash2

    def test_different_inputs_different_hashes(self):
        """Test that different inputs produce different hashes"""
        hash1 = compute_sha256("test1")
        hash2 = compute_sha256("test2")
        assert hash1 != hash2

