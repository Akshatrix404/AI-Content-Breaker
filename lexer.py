"""
lexer.py -- LEXICAL ANALYSIS PHASE
===================================
Classic compiler-style lexer: converts raw source text (the "program") into
a stream of classified TOKENS using a token-specification table (regex based
finite automaton simulation), then a second micro-pass assigns a POS
(part-of-speech) category to each WORD token via a rule-based tagger
(closed-class lexicon lookup + morphological suffix rules), analogous to how
a compiler resolves identifiers against a symbol/keyword table.
"""

import re
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# 1. TOKEN SPECIFICATION TABLE (ordered - first match wins, like lex/flex)
# ---------------------------------------------------------------------------
TOKEN_SPEC = [
    ("WHITESPACE", r"[ \t]+"),
    ("NEWLINE",    r"\n+"),
    ("NUMBER",     r"\d+(\.\d+)?"),
    ("ABBREV",     r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e)\."),
    ("ELLIPSIS",   r"\.\.\."),
    ("SENT_END",   r"[.!?]+"),
    ("COMMA",      r","),
    ("SEMICOLON",  r";"),
    ("COLON",      r":"),
    ("QUOTE",      r'["\u201c\u201d\u2018\u2019\']'),
    ("LPAREN",     r"[\(\[]"),
    ("RPAREN",     r"[\)\]]"),
    ("DASH",       r"--|\u2014|-"),
    ("WORD",       r"[A-Za-z][A-Za-z'\u2019-]*"),
    ("OTHER",      r"."),
]
MASTER_REGEX = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in TOKEN_SPEC))


@dataclass
class Token:
    type: str
    value: str
    pos: str = ""          # part-of-speech tag (filled by tagger)
    index: int = 0         # position in overall token stream
    span: tuple = (0, 0)   # (start_char, end_char) in source


def tokenize(text: str) -> List[Token]:
    """Finite-automaton-style scan of the source text into a token stream."""
    tokens = []
    idx = 0
    for m in MASTER_REGEX.finditer(text):
        kind = m.lastgroup
        val = m.group()
        if kind in ("WHITESPACE", "NEWLINE"):
            continue
        tok = Token(type=kind, value=val, index=idx, span=(m.start(), m.end()))
        tokens.append(tok)
        idx += 1
    return tokens


# ---------------------------------------------------------------------------
# 2. RULE-BASED POS TAGGER  (closed-class lexicon + morphological rules)
# ---------------------------------------------------------------------------
DETERMINERS = {"a", "an", "the", "this", "that", "these", "those", "some",
               "any", "every", "each", "no", "all", "both", "either", "neither"}

PRONOUNS = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
            "us", "them", "myself", "yourself", "himself", "herself", "itself",
            "ourselves", "themselves", "who", "whom", "whose", "which", "what",
            "this", "that", "these", "those", "one", "someone", "something",
            "anyone", "everyone", "nobody"}

PREPOSITIONS = {"in", "on", "at", "by", "for", "with", "about", "against",
                 "between", "into", "through", "during", "before", "after",
                 "above", "below", "to", "from", "up", "down", "of", "off",
                 "over", "under", "again", "further", "than", "upon", "within",
                 "without", "among", "across", "behind", "beyond", "despite",
                 "toward", "towards", "via", "per", "amid", "regarding"}

COORD_CONJ = {"and", "but", "or", "nor", "yet", "so"}
SUBORD_CONJ = {"because", "although", "though", "since", "while", "if",
               "unless", "until", "whereas", "whether", "as", "once", "wherever"}

MODALS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}
AUX_BE = {"be", "am", "is", "are", "was", "were", "been", "being"}
AUX_HAVE = {"have", "has", "had", "having"}
AUX_DO = {"do", "does", "did"}

# High-signal "AI-flavoured" discourse connectives / hedges
TRANSITION_WORDS = {
    "moreover", "furthermore", "additionally", "therefore", "consequently",
    "however", "nonetheless", "nevertheless", "thus", "hence", "overall",
    "ultimately", "importantly", "notably", "specifically", "essentially",
    "fundamentally", "significantly", "accordingly", "subsequently",
    "meanwhile", "conversely", "similarly", "likewise", "in", "conclusion",
    "summary", "additionally", "furthermore"
}
TRANSITION_PHRASES_START = {
    "in conclusion", "in summary", "to summarize", "in addition",
    "on the other hand", "as a result", "it is important to note",
    "it is worth noting", "overall,", "in essence", "in other words"
}

ADVERB_SUFFIX = re.compile(r".+ly$")
ADJ_SUFFIXES = re.compile(r".+(ful|ous|ive|able|ible|al|ic|less|ary|ent|ant)$")
VERB_SUFFIXES = re.compile(r".+(ize|ise|ate|ify|fy)$")
NOUN_SUFFIXES = re.compile(r".+(tion|sion|ment|ness|ity|ance|ence|ship|hood|dom|ism)$")

COMMON_VERBS = {
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "make", "makes", "made", "use", "uses", "used",
    "provide", "provides", "provided", "show", "shows", "showed", "help",
    "helps", "helped", "create", "creates", "created", "include", "includes",
    "included", "ensure", "ensures", "ensured", "allow", "allows", "allowed",
    "become", "becomes", "became", "give", "gives", "gave", "take", "takes",
    "took", "need", "needs", "needed", "want", "wants", "wanted", "see",
    "sees", "saw", "get", "gets", "got", "go", "goes", "went", "say", "says",
    "said", "know", "knows", "knew", "think", "thinks", "thought",
}


def _tag_word(word: str, is_first_in_sentence: bool) -> str:
    lw = word.lower()
    if lw in DETERMINERS:
        return "DET"
    if lw in PRONOUNS:
        return "PRON"
    if lw in COORD_CONJ:
        return "CCONJ"
    if lw in SUBORD_CONJ:
        return "SCONJ"
    if lw in PREPOSITIONS:
        return "PREP"
    if lw in MODALS:
        return "MODAL"
    if lw in AUX_BE or lw in AUX_HAVE or lw in AUX_DO:
        return "AUX"
    if lw in TRANSITION_WORDS:
        return "ADV_CONN"          # discourse connective adverb
    if lw in COMMON_VERBS:
        return "VERB"
    if ADVERB_SUFFIX.match(lw) and lw not in ("family", "supply", "reply", "apply"):
        return "ADV"
    if ADJ_SUFFIXES.match(lw):
        return "ADJ"
    if VERB_SUFFIXES.match(lw) or lw.endswith("ing") or lw.endswith("ed"):
        return "VERB"
    if NOUN_SUFFIXES.match(lw):
        return "NOUN"
    if not is_first_in_sentence and word[0].isupper():
        return "PROPN"
    if lw.endswith("s") and not lw.endswith("ss") and len(lw) > 3:
        return "NOUN"
    return "NOUN"  # default fallback -> majority class in English text


def tag_tokens(tokens: List[Token]) -> None:
    """In-place POS tagging pass over a token stream."""
    sentence_start = True
    for tok in tokens:
        if tok.type == "WORD":
            tok.pos = _tag_word(tok.value, sentence_start)
            sentence_start = False
        elif tok.type in ("SENT_END", "ELLIPSIS"):
            tok.pos = "SENT_END"
            sentence_start = True
        elif tok.type == "COMMA":
            tok.pos = "COMMA"
        elif tok.type == "NUMBER":
            tok.pos = "NUM"
        else:
            tok.pos = tok.type


def split_sentences(tokens: List[Token]) -> List[List[Token]]:
    """Group the flat token stream into per-sentence token lists."""
    sentences, current = [], []
    for tok in tokens:
        current.append(tok)
        if tok.pos == "SENT_END":
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return [s for s in sentences if any(t.type == "WORD" for t in s)]
