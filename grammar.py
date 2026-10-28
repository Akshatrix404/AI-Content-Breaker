"""
grammar.py -- GRAMMAR DEFINITION
================================
Defines the context-free grammar (CFG) used by parser.py. Terminals are the
POS tags produced by the lexer's tagging pass; non-terminals are syntactic
categories (S, NP, VP, PP, AP). Productions are written explicitly, and
FIRST sets are pre-computed so the parser can make LL(1)-style predictive
decisions (falling back to panic-mode recovery when no production applies).

Grammar (EBNF form actually implemented by parser.py's recursive-descent
routines -- listed here as documentation of the language being recognised):

    S    -> NP VP (CCONJ S | SCONJ S)*
    NP   -> DET? AP* (NOUN | PROPN | PRON)+ PP*
    AP   -> ADJ+
    PP   -> PREP NP
    VP   -> (MODAL | AUX)* ADV* VERB (NP)? (PP)? (ADV_CONN | ADV)*
    LEAF -> any terminal not consumed by the above (attached directly under S)
"""

NON_TERMINALS = ["S", "NP", "VP", "PP", "AP"]

# FIRST sets: which POS tags can begin each non-terminal
FIRST = {
    "NP": {"DET", "ADJ", "NOUN", "PROPN", "PRON", "NUM"},
    "AP": {"ADJ"},
    "PP": {"PREP"},
    "VP": {"MODAL", "AUX", "ADV", "VERB"},
    "S":  {"DET", "ADJ", "NOUN", "PROPN", "PRON", "NUM", "MODAL", "AUX", "ADV", "VERB"},
}

PRODUCTIONS_DOC = {
    "S": ["NP VP", "NP VP CCONJ S", "NP VP SCONJ S", "SCONJ S COMMA S"],
    "NP": ["DET AP* (NOUN|PROPN|PRON)+ PP*"],
    "AP": ["ADJ+"],
    "PP": ["PREP NP"],
    "VP": ["(MODAL|AUX)* ADV* VERB NP? PP? (ADV_CONN|ADV)*"],
}
