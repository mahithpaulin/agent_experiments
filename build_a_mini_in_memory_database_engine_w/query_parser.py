class QueryParser:
    def parse(self, query):
        """
        Parses a SQL-like query string into an intermediate representation.

        Args:
            query (str): The SQL-like query string.

        Returns:
            dict: A dictionary representing the parsed query.
        """
        tokens = query.split()
        parsed_query = {
            "select": [],
            "from": None,
            "where": None,
            "order_by": None,
            "limit": None
        }

        if "SELECT" in tokens:
            select_index = tokens.index("SELECT")
            from_index = tokens.index("FROM")
            parsed_query["select"] = tokens[select_index + 1:from_index]

        if "FROM" in tokens:
            from_index = tokens.index("FROM")
            parsed_query["from"] = tokens[from_index + 1]

        if "WHERE" in tokens:
            where_index = tokens.index("WHERE")
            order_by_index = tokens.index("ORDER BY") if "ORDER BY" in tokens else len(tokens)
            parsed_query["where"] = " ".join(tokens[where_index + 1:order_by_index])

        if "ORDER BY" in tokens:
            order_by_index = tokens.index("ORDER BY")
            limit_index = tokens.index("LIMIT") if "LIMIT" in tokens else len(tokens)
            parsed_query["order_by"] = tokens[order_by_index + 2:limit_index]

        if "LIMIT" in tokens:
            limit_index = tokens.index("LIMIT")
            parsed_query["limit"] = int(tokens[limit_index + 1])

        return parsed_query