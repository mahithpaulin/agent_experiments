class BTreeIndex:
    def __init__(self):
        self.index = {}

    def add(self, key, row_id):
        """
        Adds a key and its corresponding row ID to the B-tree index.

        Args:
            key: The key to index.
            row_id: The row ID associated with the key.
        """
        if key not in self.index:
            self.index[key] = []
        self.index[key].append(row_id)

    def get(self, key):
        """
        Retrieves the row IDs associated with a key.

        Args:
            key: The key to look up.

        Returns:
            list: A list of row IDs associated with the key.
        """
        return self.index.get(key, [])