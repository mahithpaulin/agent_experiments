class QueryPlanner:
    def __init__(self, indexes):
        """
        Initializes the QueryPlanner with available indexes.

        Args:
            indexes (dict): A dictionary of available indexes for tables.
        """
        self.indexes = indexes

    def plan(self, query):
        """
        Plans the execution strategy for a query.

        Args:
            query (dict): The parsed query.

        Returns:
            str: The execution plan ("full_scan" or "index_scan").
        """
        if query["where"] and query["where"] in self.indexes:
            return "index_scan"
        return "full_scan"