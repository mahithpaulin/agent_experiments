import json
import math
import os

import numpy as np
import pytest

# Import the classes from the main module
from main import CORPUS, IndexBuilder, TfIdfScorer, SearchEngine


@pytest.fixture(scope="module")
def engine():
    """Create a SearchEngine instance once for all tests."""
    return SearchEngine(CORPUS)


def test_inverted_index_postings(engine):
    """Verify that the inverted index contains correct postings for selected terms."""
    idx = engine.index_builder

    # Terms that appear in exactly one document
    single_doc_terms = {
        "machine": {1: 1},
        "learning": {1: 1},
        "artificial": {1: 1},
        "intelligence": {1: 1},
        "renewable": {3: 1},
        "energy": {3: 1},
        "climate": {3: 1},
        "healthcare": {8: 1},
    }

    for term, expected_postings in single_doc_terms.items():
        assert term in idx.inverted_index, f"Term '{term}' missing from inverted index"
        assert idx.inverted_index[term] == expected_postings, (
            f"Postings for term '{term}' incorrect: "
            f"got {idx.inverted_index[term]}, expected {expected_postings}"
        )

    # A common stop‑word that should appear in many documents
    assert "the" in idx.inverted_index
    assert len(idx.inverted_index["the"]) > 1, "Term 'the' should be present in multiple docs"


def test_tf_idf_idf_values(engine):
    """Check that the IDF values for terms with document frequency 1 equal log(N)."""
    scorer = engine.scorer
    N = len(CORPUS)
    expected_idf = math.log(N / 1)  # log(20)

    # Choose a few terms with df = 1
    for term in ["machine", "renewable", "climate"]:
        idx = scorer.term_to_idx[term]
        idf_val = scorer.idf_vector[idx]
        np.testing.assert_almost_equal(
            idf_val, expected_idf, decimal=6,
            err_msg=f"IDF for term '{term}' incorrect (got {idf_val}, expected {expected_idf})"
        )

    # Verify that a highly frequent term gets a smaller IDF
    the_idx = scorer.term_to_idx["the"]
    the_idf = scorer.idf_vector[the_idx]
    assert the_idf < expected_idf, "Common term 'the' should have smaller IDF than rare terms"


def test_document_vectors_tf_idf(engine):
    """Ensure that document vectors contain TF‑IDF values for known terms."""
    scorer = engine.scorer
    N = len(CORPUS)

    # Document 1 contains the term 'machine' once
    doc_id = 1
    term = "machine"
    term_idx = scorer.term_to_idx[term]
    tf_idf_val = scorer.doc_vectors[doc_id, term_idx]
    expected = 1 * math.log(N / 1)  # TF * IDF
    np.testing.assert_almost_equal(tf_idf_val, expected, decimal=6)

    # Document 3 contains the term 'energy' once
    doc_id = 3
    term = "energy"
    term_idx = scorer.term_to_idx[term]
    tf_idf_val = scorer.doc_vectors[doc_id, term_idx]
    expected = 1 * math.log(N / 1)
    np.testing.assert_almost_equal(tf_idf_val, expected, decimal=6)


def test_ranking_order_for_queries(engine):
    """Validate that the ranking order for each hard‑coded query matches expectations."""
    scorer = engine.scorer

    # Query 1: machine learning algorithms
    query1 = "machine learning algorithms"
    ranking1 = scorer.rank_documents(query1)
    top_doc1 = ranking1[0][0]
    assert top_doc1 == 1, f"Top document for query1 should be doc 1, got {top_doc1}"

    # Query 2: renewable energy climate
    query2 = "renewable energy climate"
    ranking2 = scorer.rank_documents(query2)
    top_doc2 = ranking2[0][0]
    assert top_doc2 == 3, f"Top document for query2 should be doc 3, got {top_doc2}"

    # Query 3: artificial intelligence and healthcare
    query3 = "artificial intelligence and healthcare"
    ranking3 = scorer.rank_documents(query3)

    # Expect doc 1 (contains artificial & intelligence) to outrank doc 8 (contains healthcare)
    assert ranking3[0][0] == 1, f"First document for query3 should be doc 1, got {ranking3[0][0]}"
    # Find doc 8 in the ranking and ensure it appears after doc 1
    doc8_position = next(i for i, (doc_id, _) in enumerate(ranking3) if doc_id == 8)
    assert doc8_position > 0, "Doc 8 should appear in the ranking for query3"
    assert ranking3[0][0] != 8, "Doc 8 should not be ranked above doc 1 for query3"


def test_results_json_structure(tmp_path, engine):
    """Run the full search and verify that results.json is correctly formed."""
    queries = [
        "machine learning algorithms",
        "renewable energy climate",
        "artificial intelligence and healthcare"
    ]
    results = engine.search(queries, top_k=3)

    # Write to a temporary file
    output_file = tmp_path / "results.json"
    engine.save_results(results, str(output_file))

    # Load and inspect the JSON
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # The JSON should have exactly three top‑level keys (one per query)
    assert set(data.keys()) == set(queries)

    # Each entry should be a list of dictionaries with 'doc_id' and 'score'
    for q in queries:
        entries = data[q]
        assert isinstance(entries, list)
        assert all(isinstance(item, dict) for item in entries)
        assert all("doc_id" in item and "score" in item for item in entries)

    # Verify that the top‑ranked doc IDs match the expectations from earlier tests
    assert data[queries[0]][0]["doc_id"] == 1
    assert data[queries[1]][0]["doc_id"] == 3
    assert data[queries[2]][0]["doc_id"] == 1