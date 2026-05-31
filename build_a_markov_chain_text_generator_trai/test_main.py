import pytest
from main import generate_sentences


def test_sentence_count_and_length():
    sentences = generate_sentences(num_sentences=3, min_words=8)
    assert len(sentences) == 3
    for s in sentences:
        words = s.split()
        assert len(words) >= 8
        assert s[-1] in ".!?"  # sentence must end with terminal punctuation


def test_deterministic_output():
    first = generate_sentences(num_sentences=1, min_words=8)
    second = generate_sentences(num_sentences=1, min_words=8)
    assert first == second  # same seed → identical result


def test_min_words_edge():
    sentences = generate_sentences(num_sentences=1, min_words=1)
    assert len(sentences) == 1
    assert sentences[0][-1] in ".!?"  # still ends with punctuation
    # ensure at least one word is present
    assert len(sentences[0].split()) >= 1
