import json
import re
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple


# ----------------------------------------------------------------------
# Corpus (20 documents, each 3‑5 sentences)
# ----------------------------------------------------------------------
CORPUS: List[str] = [
    "The quick brown fox jumps over the lazy dog. It is a classic example sentence. Many people use it for typing practice.",
    "Artificial intelligence and machine learning are rapidly evolving fields. Researchers develop new algorithms each year. These technologies impact many industries.",
    "Python is a versatile programming language. It is widely used for web development, data analysis, and automation. Its syntax is clear and readable.",
    "Climate change poses significant challenges to humanity. Renewable energy sources are essential for a sustainable future. Governments invest in green technologies.",
    "The human brain contains billions of neurons. It processes information faster than any computer. Understanding cognition is a major scientific goal.",
    "Space exploration has led to many discoveries. Satellites provide critical data for weather forecasting. The International Space Station orbits Earth constantly.",
    "Economics studies the allocation of scarce resources. Markets respond to supply and demand dynamics. Fiscal policies influence national growth.",
    "Literature reflects cultural values and historical contexts. Classic novels often explore human nature. Poetry captures emotions in concise forms.",
    "Healthcare systems aim to improve patient outcomes. Vaccines prevent the spread of infectious diseases. Medical research advances treatments.",
    "Traveling expands one's perspective. Different cuisines showcase regional traditions. Language barriers can be overcome with technology.",
    "Education fosters critical thinking and innovation. Schools adapt curricula to modern needs. Online training platforms increase accessibility.",
    "Sports promote physical fitness and teamwork. International competitions unite nations. Fans passionately support their favorite teams.",
    "Music influences mood and creativity. Instruments produce diverse sounds across genres. Concerts bring audiences together.",
    "Technology drives economic development. Startups introduce disruptive ideas. Venture capital funds fuel growth.",
    "History records the rise and fall of civilizations. Archaeological sites reveal ancient lifestyles. Preservation protects cultural heritage.",
    "Food safety regulations ensure public health. Restaurants adhere to hygiene standards. Consumers trust quality certifications.",
    "Transportation networks connect cities and regions. Public transit reduces traffic congestion. Autonomous vehicles are emerging technologies.",
    "Environmental conservation protects biodiversity. Protected areas safeguard habitats. Community initiatives promote sustainable practices.",
    "Finance manages assets, liabilities, and risk. Stock markets reflect investor sentiment. Banking institutions provide essential services.",
    "Arts inspire imagination and expression. Museums exhibit masterpieces for public appreciation. Creative workshops nurture talent."
]


# ----------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    """Lower‑case and split on non‑alphanumeric characters."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    return tokens


# ----------------------------------------------------------------------
# Index Builder
# ----------------------------------------------------------------------
class IndexBuilder:
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.inverted_index: Dict[str, Dict[int, int]] = defaultdict(dict)
        self.doc_freq: Dict[str, int] = {}
        self._build_index()

    def _build_index(self) -> None:
        for doc_id, text in enumerate(self.documents):
            term_counts: Dict[str, int] = defaultdict(int)
            for term in tokenize(text):
                term_counts[term] += 1
            for term, freq in term_counts.items():
                self.inverted_index[term][doc_id] = freq

        # document frequency (df) for each term
        self.doc_freq = {term: len(postings) for term, postings in self.inverted_index.items()}


# ----------------------------------------------------------------------
# TF‑IDF Scorer
# ----------------------------------------------------------------------
class TfIdfScorer:
    def __init__(self, index_builder: IndexBuilder):
        self.index = index_builder
        self.N = len(self.index.documents)
        self.vocab: List[str] = sorted(self.index.inverted_index.keys())
        self.term_to_idx: Dict[str, int] = {term: i for i, term in enumerate(self.vocab)}
        self.idf_vector: np.ndarray = self._compute_idf()
        self.doc_vectors: np.ndarray = self._compute_doc_vectors()

    def _compute_idf(self) -> np.ndarray:
        df = np.array([self.index.doc_freq[term] for term in self.vocab], dtype=float)
        # Prevent division by zero (should not happen)
        df = np.where(df == 0, 1.0, df)
        idf = np.log(self.N / df)
        return idf

    def _compute_doc_vectors(self) -> np.ndarray:
        # shape: (num_docs, vocab_size)
        vectors = np.zeros((self.N, len(self.vocab)), dtype=float)
        for term, postings in self.index.inverted_index.items():
            term_idx = self.term_to_idx[term]
            for doc_id, tf in postings.items():
                vectors[doc_id, term_idx] = tf
        # TF‑IDF = TF * IDF (broadcast)
        vectors *= self.idf_vector
        return vectors

    def _vectorize_query(self, query: str) -> np.ndarray:
        q_vec = np.zeros(len(self.vocab), dtype=float)
        for term in tokenize(query):
            if term in self.term_to_idx:
                idx = self.term_to_idx[term]
                # raw term frequency in the query
                q_vec[idx] += 1
        q_vec *= self.idf_vector
        return q_vec

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def rank_documents(self, query: str) -> List[Tuple[int, float]]:
        """Return list of (doc_id, similarity) sorted by descending similarity then doc_id."""
        q_vec = self._vectorize_query(query)
        scores = [
            (doc_id, self._cosine_similarity(q_vec, self.doc_vectors[doc_id]))
            for doc_id in range(self.N)
        ]
        scores.sort(key=lambda x: (-x[1], x[0]))
        return scores


# ----------------------------------------------------------------------
# Search Engine
# ----------------------------------------------------------------------
class SearchEngine:
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.index_builder = IndexBuilder(documents)
        self.scorer = TfIdfScorer(self.index_builder)

    def search(self, queries: List[str], top_k: int = 3) -> Dict[str, List[Dict]]:
        """
        For each query, compute rankings, print top_k results,
        and store the full ranking in a dictionary.
        """
        results: Dict[str, List[Dict]] = {}
        for query in queries:
            ranking = self.scorer.rank_documents(query)
            # Store full ranking for JSON output
            results[query] = [
                {"doc_id": doc_id, "score": round(score, 6)} for doc_id, score in ranking
            ]

            # Print top_k
            print(f"\nQuery: \"{query}\"")
            for rank, (doc_id, score) in enumerate(ranking[:top_k], start=1):
                print(f"  Rank {rank}: Doc {doc_id} (Score: {score:.4f})")
        return results

    def save_results(self, results: Dict[str, List[Dict]], filename: str = "results.json") -> None:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define three hard‑coded queries
    queries = [
        "machine learning algorithms",
        "renewable energy climate",
        "artificial intelligence and healthcare"
    ]

    engine = SearchEngine(CORPUS)
    full_results = engine.search(queries, top_k=3)
    engine.save_results(full_results, "results.json")