"""
Unit tests for FAISS embedding functionality
"""

import pytest
from suql.faiss_embedding import (
    compute_sha256,
    consistent_tuple_hash,
    OrderedSet,
)


class TestComputeSha256:
    """Test cases for compute_sha256 function in faiss_embedding"""

    def test_basic_hash(self):
        """Test SHA256 hash computation"""
        text = "test embedding"
        result = compute_sha256(text)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_consistency(self):
        """Test that same input produces same hash"""
        text = "consistent embedding test"
        hash1 = compute_sha256(text)
        hash2 = compute_sha256(text)
        assert hash1 == hash2


class TestConsistentTupleHash:
    """Test cases for consistent_tuple_hash function"""

    def test_simple_tuple(self):
        """Test hashing a simple tuple"""
        test_tuple = ("table", "column", "value")
        result = consistent_tuple_hash(test_tuple)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_tuple_hash_consistency(self):
        """Test that same tuple produces same hash"""
        test_tuple = ("restaurants", "reviews", "test")
        hash1 = consistent_tuple_hash(test_tuple)
        hash2 = consistent_tuple_hash(test_tuple)
        assert hash1 == hash2

    def test_different_tuples_different_hashes(self):
        """Test that different tuples produce different hashes"""
        tuple1 = ("table1", "col1")
        tuple2 = ("table2", "col2")
        hash1 = consistent_tuple_hash(tuple1)
        hash2 = consistent_tuple_hash(tuple2)
        assert hash1 != hash2


class TestOrderedSet:
    """Test cases for OrderedSet class"""

    def test_empty_ordered_set(self):
        """Test creating an empty OrderedSet"""
        ordered_set = OrderedSet()
        assert len(ordered_set) == 0

    def test_add_items(self):
        """Test adding items to OrderedSet"""
        ordered_set = OrderedSet()
        ordered_set.add("item1")
        ordered_set.add("item2")
        assert len(ordered_set) == 2
        assert "item1" in ordered_set
        assert "item2" in ordered_set

    def test_no_duplicates(self):
        """Test that OrderedSet doesn't allow duplicates"""
        ordered_set = OrderedSet()
        ordered_set.add("item1")
        ordered_set.add("item1")
        assert len(ordered_set) == 1

    def test_preserves_order(self):
        """Test that OrderedSet preserves insertion order"""
        ordered_set = OrderedSet()
        items = ["first", "second", "third"]
        for item in items:
            ordered_set.add(item)
        
        result = list(ordered_set)
        assert result == items

    def test_union(self):
        """Test union operation"""
        set1 = OrderedSet(["a", "b"])
        set2 = OrderedSet(["b", "c"])
        union = set1.union(set2)
        assert len(union) == 3
        assert "a" in union
        assert "b" in union
        assert "c" in union

    def test_init_with_iterable(self):
        """Test initializing OrderedSet with iterable"""
        items = ["item1", "item2", "item3"]
        ordered_set = OrderedSet(items)
        assert len(ordered_set) == 3
        assert all(item in ordered_set for item in items)

