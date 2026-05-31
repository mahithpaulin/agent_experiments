# main.py
class _Node:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    """
    Least Recently Used (LRU) cache with O(1) get and put operations.
    """

    def __init__(self, capacity: int):
        """
        Initialize the LRU cache with a positive capacity.
        If capacity is non‑positive, the cache behaves as a no‑op cache.
        """
        self.capacity = max(0, capacity)
        self.cache = {}  # key -> _Node
        # Dummy head and tail to avoid edge checks
        self.head = _Node(0, 0)  # Most recently used after head
        self.tail = _Node(0, 0)  # Least recently used before tail
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0

    def _add_to_head(self, node: _Node) -> None:
        """Insert node right after head (most recent position)."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: _Node) -> None:
        """Disconnect node from the linked list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _move_to_head(self, node: _Node) -> None:
        """Move an existing node to the head (most recent)."""
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> _Node:
        """Remove and return the least recently used node (just before tail)."""
        lru = self.tail.prev
        self._remove_node(lru)
        return lru

    def get(self, key: int) -> int:
        """
        Return the value of the key if present, otherwise -1.
        Access updates the usage order.
        """
        node = self.cache.get(key)
        if not node:
            return -1
        # Move the accessed node to the head (most recent)
        self._move_to_head(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        """
        Insert or update the value of the key.
        If inserting exceeds capacity, evicts the least recently used key.
        """
        if self.capacity == 0:
            return  # No‑op cache

        node = self.cache.get(key)
        if node:
            # Update existing node and move to head
            node.value = value
            self._move_to_head(node)
        else:
            # Create new node
            new_node = _Node(key, value)
            self.cache[key] = new_node
            self._add_to_head(new_node)
            self.size += 1

            if self.size > self.capacity:
                # Evict LRU
                tail_node = self._pop_tail()
                del self.cache[tail_node.key]
                self.size -= 1


# test_main.py
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