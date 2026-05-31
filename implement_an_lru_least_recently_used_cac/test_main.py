import pytest
from main import LRUCache


def test_basic_put_and_get():
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1  # returns 1
    cache.put(3, 3)           # evicts key 2
    assert cache.get(2) == -1  # returns -1 (not found)
    cache.put(4, 4)           # evicts key 1
    assert cache.get(1) == -1  # returns -1 (not found)
    assert cache.get(3) == 3   # returns 3
    assert cache.get(4) == 4   # returns 4


def test_access_updates_order():
    cache = LRUCache(3)
    cache.put(1, 10)
    cache.put(2, 20)
    cache.put(3, 30)
    # Access key 2 to make it most recent
    assert cache.get(2) == 20
    # Insert new key, should evict key 1 (least recent)
    cache.put(4, 40)
    assert cache.get(1) == -1
    assert cache.get(2) == 20
    assert cache.get(3) == 30
    assert cache.get(4) == 40


def test_overwrite_existing_key():
    cache = LRUCache(2)
    cache.put(1, 100)
    cache.put(2, 200)
    # Overwrite key 1 with new value
    cache.put(1, 101)
    assert cache.get(1) == 101
    # Now key 2 should be LRU; inserting new key evicts key 2
    cache.put(3, 300)
    assert cache.get(2) == -1
    assert cache.get(1) == 101
    assert cache.get(3) == 300


def test_zero_capacity_cache():
    cache = LRUCache(0)
    cache.put(1, 1)
    assert cache.get(1) == -1
    cache.put(2, 2)
    assert cache.get(2) == -1


def test_multiple_evictions():
    cache = LRUCache(1)
    cache.put(10, 10)
    assert cache.get(10) == 10
    cache.put(20, 20)  # evicts 10
    assert cache.get(10) == -1
    assert cache.get(20) == 20
    cache.put(30, 30)  # evicts 20
    assert cache.get(20) == -1
    assert cache.get(30) == 30


if __name__ == "__main__":
    pytest.main([__file__])