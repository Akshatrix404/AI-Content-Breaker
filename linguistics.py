"""
linguistics.py -- EXTENDED LEXICAL & STATISTICAL ANALYSIS PHASE
===================================================================
A second attribute pass that runs alongside the syntax-tree attribute
grammar, but operates over the flat token/tag stream and the raw text
rather than the tree -- the compiler-design analogue of a peephole /
statistics pass that runs after the main semantic pass. Everything here is
still fully deterministic and explainable: no ML model, just information
theory (Shannon entropy, DEFLATE compression ratio) and closed-class
lexicon lookups (stopwords, hedges, contractions, pronoun person).

Two kinds of output:
  * SCORED features   -> feed scorer.py's weighted report
  * INFORMATIONAL metrics -> shown to the user for insight, but deliberately
    NOT weighted into the score, because on their own they are not reliable
    AI-vs-human discriminators (e.g. readability grade level says nothing
    about authorship). Keeping them out keeps the score defensible.
"""

import math
import re
import zlib
import statistics as stats
from collections import Counter

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "to",
    "in", "on", "at", "by", "for", "with", "about", "as", "into", "like",
    "through", "after", "over", "between", "out", "against", "during",
    "without", "before", "under", "around", "among", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "this", "that", "these", "those", "i", "you", "he", "she",
    "it", "we", "they", "them", "his", "her", "its", "our", "their",
    "not", "no", "yes", "there", "here", "what", "which", "who", "whom",
}

FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
SECOND_PERSON = {"you", "your", "yours", "yourself", "yourselves"}
THIRD_PERSON = {"he", "him", "his", "himself", "she", "her", "hers", "herself",
                 "it", "its", "itself", "they", "them", "their", "theirs", "themselves"}

HEDGE_WORDS = {
    "perhaps", "maybe", "possibly", "likely", "probably", "seem", "seems",
    "seemed", "appears", "appear", "appeared", "suggest", "suggests",
    "suggested", "indicate", "indicates", "might", "could", "may", "tend",
    "tends", "generally", "often", "typically", "arguably", "somewhat",
    "relatively", "roughly", "presumably",
}

CONTRACTION_PATTERN = re.compile(r"[A-Za-z]+['\u2019][A-Za-z]+")

VOWEL_GROUPS = re.compile(r"[aeiouyAEIOUY]+")


def count_syllables(word: str) -> int:
    """Heuristic syllable counter: count vowel-cluster groups, adjust for
    a silent trailing 'e'. Not phonetically perfect but standard practice
    for readability formulas implemented without a pronunciation dictionary."""
    w = word.lower()
    groups = VOWEL_GROUPS.findall(w)
    n = len(groups)
    if w.endswith("e") and not w.endswith("le") and n > 1:
        n -= 1
    return max(1, n)


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counter.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def analyze_linguistics(text: str, word_tokens_per_sentence, pos_tag_sequence,
                          punctuation_counts, sentence_texts) -> dict:
    """
    word_tokens_per_sentence: list[list[Token]] (WORD tokens only, per sentence)
    pos_tag_sequence: flat list[str] of every POS tag in document order
    sentence_texts: list[str] plain sentence strings (for similarity check)
    """
    all_words = [t.value.lower() for sent in word_tokens_per_sentence for t in sent]
    n_words = len(all_words)
    n_sentences = max(1, len(word_tokens_per_sentence))

    # -- readability (informational only) ------------------------------------
    total_syllables = sum(count_syllables(w) for w in all_words)
    avg_sentence_len = n_words / n_sentences
    avg_syllables_per_word = total_syllables / n_words if n_words else 0
    flesch_ease = 206.835 - 1.015 * avg_sentence_len - 84.6 * avg_syllables_per_word if n_words else 0
    flesch_kincaid_grade = 0.39 * avg_sentence_len + 11.8 * avg_syllables_per_word - 15.59 if n_words else 0
    avg_word_len = (sum(len(w) for w in all_words) / n_words) if n_words else 0
    long_word_ratio = (sum(1 for w in all_words if len(w) >= 7) / n_words) if n_words else 0

    # -- pronoun person distribution (informational) --------------------------
    fp = sum(1 for w in all_words if w in FIRST_PERSON)
    sp = sum(1 for w in all_words if w in SECOND_PERSON)
    tp = sum(1 for w in all_words if w in THIRD_PERSON)
    pronoun_total = fp + sp + tp

    # -- stopword ratio (informational) ---------------------------------------
    stopword_ratio = (sum(1 for w in all_words if w in STOPWORDS) / n_words) if n_words else 0

    # -- hedge / modal density (informational) ---------------------------------
    hedge_count = sum(1 for w in all_words if w in HEDGE_WORDS)
    hedge_per_100 = hedge_count / n_words * 100 if n_words else 0

    # -- SCORED: contraction scarcity ------------------------------------------
    contraction_count = sum(1 for w in all_words if CONTRACTION_PATTERN.fullmatch(w))
    contraction_per_100 = contraction_count / n_words * 100 if n_words else 0
    # human/informal writing uses contractions; scarce contractions -> more AI-like
    contraction_scarcity_score = _clip(1 - contraction_per_100 / 1.2)

    # -- SCORED: POS bigram entropy (predictability of syntax stream) ---------
    tag_bigrams = Counter(zip(pos_tag_sequence, pos_tag_sequence[1:]))
    bigram_entropy = _entropy(tag_bigrams)
    distinct_tags = len(set(pos_tag_sequence)) or 1
    max_entropy = math.log2(max(2, distinct_tags * distinct_tags))
    normalized_entropy = bigram_entropy / max_entropy if max_entropy else 0
    pos_predictability_score = _clip(1 - normalized_entropy / 0.85)

    # -- SCORED: compression ratio (DEFLATE) as redundancy proxy --------------
    raw_bytes = text.encode("utf-8", errors="ignore")
    if len(raw_bytes) > 80:
        compressed = zlib.compress(raw_bytes, level=9)
        compression_ratio = len(compressed) / len(raw_bytes)
        compression_score = _clip((0.52 - compression_ratio) / 0.18)
        compression_note = f"DEFLATE compresses this text to {compression_ratio*100:.1f}% of its original size."
    else:
        compression_ratio = None
        compression_score = 0.5
        compression_note = "Text too short for a reliable compression-ratio signal."

    # -- SCORED: consecutive-sentence similarity (templated phrasing) ---------
    def word_set(s):
        return {w.lower() for w in re.findall(r"[A-Za-z']+", s) if w.lower() not in STOPWORDS}

    sims = []
    for a, b in zip(sentence_texts, sentence_texts[1:]):
        wa, wb = word_set(a), word_set(b)
        if not wa or not wb:
            continue
        j = len(wa & wb) / len(wa | wb)
        sims.append(j)
    avg_similarity = stats.mean(sims) if sims else None
    similarity_score = _clip((avg_similarity - 0.03) / 0.22) if avg_similarity is not None else 0.5

    # -- SCORED: passive voice ratio (soft signal, small weight) --------------
    # filled in by pipeline.py from parser-derived voice flags; placeholder here
    return {
        "informational": {
            "avg_sentence_length_words": round(avg_sentence_len, 2),
            "avg_word_length_chars": round(avg_word_len, 2),
            "avg_syllables_per_word": round(avg_syllables_per_word, 2),
            "flesch_reading_ease": round(flesch_ease, 1),
            "flesch_kincaid_grade": round(flesch_kincaid_grade, 1),
            "long_word_ratio_pct": round(long_word_ratio * 100, 1),
            "stopword_ratio_pct": round(stopword_ratio * 100, 1),
            "hedge_words_per_100": round(hedge_per_100, 2),
            "pronoun_person_pct": {
                "first": round(fp / pronoun_total * 100, 1) if pronoun_total else 0,
                "second": round(sp / pronoun_total * 100, 1) if pronoun_total else 0,
                "third": round(tp / pronoun_total * 100, 1) if pronoun_total else 0,
            },
            "contractions_per_100_words": round(contraction_per_100, 2),
            "pos_bigram_entropy_bits": round(bigram_entropy, 2),
            "compression_ratio_pct": round(compression_ratio * 100, 1) if compression_ratio else None,
        },
        "scored": {
            "contraction_scarcity": {
                "score": contraction_scarcity_score,
                "note": f"{contraction_per_100:.2f} contractions per 100 words (e.g. don't, it's)",
                "label": "Contraction Scarcity",
            },
            "pos_bigram_predictability": {
                "score": pos_predictability_score,
                "note": f"POS-tag bigram entropy = {bigram_entropy:.2f} bits "
                        f"({normalized_entropy*100:.0f}% of max) -- lower means more predictable syntax.",
                "label": "Syntactic Predictability (POS-entropy)",
            },
            "compression_predictability": {
                "score": compression_score,
                "note": compression_note,
                "label": "Text Redundancy (compression ratio)",
            },
            "sentence_similarity_repetition": {
                "score": similarity_score,
                "note": (f"Average word-overlap between consecutive sentences = {avg_similarity*100:.0f}%."
                          if avg_similarity is not None else "Not enough sentences to compare."),
                "label": "Consecutive-Sentence Similarity",
            },
        }
    }
