import pytest
from database import Database
from query_parser import QueryParser
from indexing import BTreeIndex
from query_planner import QueryPlanner

def test_database_operations():
    # Initialize the database
    db = Database()
    db.create_table("employees", ["id", "name", "age", "department", "salary"])
    db.insert("employees", [1, "Alice", 30, "HR", 50000])
    db.insert("employees", [2, "Bob", 25, "Engineering", 60000])
    db.insert("employees", [3, "Charlie", 35, "HR", 55000])

    # Query the database
    query = "SELECT name, age FROM employees WHERE department = 'HR' ORDER BY age LIMIT 1"
    results = db.query(query)

    # Assert the results
    assert results == [["Alice", 30]]