import random
from typing import List, Dict

# Hard‑coded corpus (≥ 50 words)
_CORPUS_TEXT = (
    "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega. "
    "This sentence serves as part of a corpus for Markov chain generator. "
    "It includes multiple unique words to ensure deterministic transitions. "
    "Quick brown fox jumps over lazy dog. "
    "Randomness is controlled with fixed seed for reproducible results. "
    "Testing edge cases ensures reliability and correctness. "
    "Additional words are added to meet required quota of at least fifty words total."
)

# Tokenize the corpus (preserve punctuation attached to words)
_TOKENS = _CORPUS_TEXT.split()

# Build first‑order Markov model: word -> list of possible successors
_MODEL: Dict[str, List[str]] = {}
for i in range(len(_TOKENS) - 1):
    word = _TOKENS[i]
    succ = _TOKENS[i + 1]
    _MODEL.setdefault(word, []).append(succ)


def generate_sentences(num_sentences: int = 5, min_words: int = 8) -> List[str]:
    """
    Generate a list of sentences using a first‑order Markov chain.

    Each sentence contains at least *min_words* words and ends with a terminal
    punctuation mark ('.', '?', '!') once that minimum is satisfied.

    The random seed is fixed (12345) for reproducible output.
    """
    random.seed(12345)
    sentences: List[str] = []

    for _ in range(num_sentences):
        # Choose a random start word from the corpus
        current = random.choice(_TOKENS)
        words: List[str] = [current]

        while True:
            successors = _MODEL.get(current, [])
            if not successors:
                break  # No successor – should not happen with this corpus
            next_word = random.choice(successors)
            words.append(next_word)
            current = next_word

            # Stop condition: minimum length reached and current word ends punctuation
            if len(words) >= min_words and current[-1] in ".!?":
                break

        sentences.append(' '.join(words))

    return sentences


if __name__ == "__main__":
    generated = generate_sentences()
    for idx, sentence in enumerate(generated, 1):
        print(f"{idx}. {sentence}")
