"""
symbol_table.py -- SYMBOL TABLE CONSTRUCTION PHASE
=====================================================
A compiler's symbol table records every identifier it has seen, along with
metadata (scope, type, usage count). Here the "identifiers" of the document
are: words, bigrams, sentence-opening words, and parse-tree rule-signatures.
Building this table is what lets semantic.py later detect *repetition* --
the single strongest structural signal used throughout this framework,
since large-language-model output tends to reuse a narrower set of syntactic
"shapes" and discourse connectives than naturally-written text.
"""

from collections import Counter, defaultdict
from lexer import TRANSITION_WORDS, TRANSITION_PHRASES_START


class SymbolTable:
    def __init__(self):
        self.word_freq = Counter()
        self.bigram_freq = Counter()
        self.opener_freq = Counter()
        self.rule_sig_freq = Counter()
        self.transition_hits = []          # list of (sentence_idx, word)
        self.sentence_lengths = []         # word count per sentence
        self.sentence_depths = []          # parse-tree depth per sentence
        self.total_words = 0

    def add_sentence(self, sentence_idx, word_tokens, opener_word, rule_sigs, tree_depth):
        words = [t.value.lower() for t in word_tokens]
        self.word_freq.update(words)
        self.total_words += len(words)
        self.sentence_lengths.append(len(words))
        self.sentence_depths.append(tree_depth)

        for a, b in zip(words, words[1:]):
            self.bigram_freq[(a, b)] += 1

        if opener_word:
            self.opener_freq[opener_word.lower()] += 1

        for w in words:
            if w in TRANSITION_WORDS:
                self.transition_hits.append((sentence_idx, w))

        for sig in rule_sigs:
            self.rule_sig_freq[sig] += 1

    # -- derived statistics used by scorer.py --------------------------------
    def type_token_ratio(self):
        if self.total_words == 0:
            return 0.0
        return len(self.word_freq) / self.total_words

    def hapax_ratio(self):
        if not self.word_freq:
            return 0.0
        hapax = sum(1 for c in self.word_freq.values() if c == 1)
        return hapax / len(self.word_freq)

    def top_repeated_rule_signatures(self, n=5):
        return self.rule_sig_freq.most_common(n)

    def top_repeated_openers(self, n=5):
        return [(w, c) for w, c in self.opener_freq.most_common(n) if c > 1]

    def to_dict(self):
        return {
            "vocab_size": len(self.word_freq),
            "total_words": self.total_words,
            "type_token_ratio": round(self.type_token_ratio(), 4),
            "hapax_ratio": round(self.hapax_ratio(), 4),
            "top_words": self.word_freq.most_common(10),
            "top_bigrams": [(f"{a} {b}", c) for (a, b), c in self.bigram_freq.most_common(8)],
            "top_rule_signatures": self.top_repeated_rule_signatures(6),
            "repeated_openers": self.top_repeated_openers(6),
            "transition_word_count": len(self.transition_hits),
        }
