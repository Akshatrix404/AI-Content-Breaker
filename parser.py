"""
parser.py -- SYNTAX ANALYSIS PHASE
===================================
A predictive recursive-descent parser driven by the CFG in grammar.py.
Input: one sentence's list of lexer.Token (with .pos already tagged).
Output: a Node tree (the parse / syntax tree) rooted at 'S'.

Implements standard compiler-construction machinery:
  - lookahead-based production selection (via FIRST sets)
  - panic-mode error recovery: if no production matches the current
    lookahead, the offending token is wrapped as a LEAF node and consumption
    continues, so a single malformed span never aborts the whole parse
    (mirrors how real compilers recover from a syntax error and keep going
    so later errors can still be reported).
"""

from dataclasses import dataclass, field
from typing import List, Optional
from grammar import FIRST


@dataclass
class Node:
    symbol: str                 # non-terminal name, POS tag, or "LEAF"
    children: list = field(default_factory=list)
    token: object = None        # set only for terminal/leaf nodes
    attrs: dict = field(default_factory=dict)   # filled in by attributes.py

    def is_leaf(self):
        return not self.children


class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens):
        # only WORD-derived / content tokens matter to the grammar;
        # punctuation terminals (COMMA, SENT_END, etc.) are still tracked
        # positionally so nothing from the sentence is silently dropped.
        self.tokens = tokens
        self.i = 0
        self.n = len(tokens)
        self.recoveries = 0   # count of panic-mode recoveries (a diagnostic signal)

    # -- low level stream helpers -------------------------------------------------
    def peek(self, offset=0):
        j = self.i + offset
        return self.tokens[j] if j < self.n else None

    def advance(self):
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def pos_at(self, offset=0):
        t = self.peek(offset)
        return t.pos if t else None

    # -- grammar rules --------------------------------------------------------
    def parse_sentence(self) -> Node:
        root = Node("S")
        # fronted discourse adverb, e.g. "Moreover, ..." / "However, ..."
        while self.i < self.n and self.pos_at() == "ADV_CONN":
            root.children.append(Node("ADV_CONN", token=self.advance()))
            if self.pos_at() == "COMMA":
                root.children.append(Node("COMMA", token=self.advance()))
        if self.i < self.n and self.pos_at() in FIRST["NP"]:
            root.children.append(self.parse_np())
        if self.i < self.n and self.pos_at() in FIRST["VP"]:
            root.children.append(self.parse_vp())
        # handle coordinated / subordinated clause continuation
        while self.i < self.n and self.pos_at() in ("CCONJ", "SCONJ"):
            conj_tok = self.advance()
            root.children.append(Node(self.pos_at(-1) or "CONJ", token=conj_tok))
            if self.i < self.n:
                root.children.append(self.parse_sentence())
            break
        # anything left un-consumed: attach as recovered leaves (panic mode).
        # Trailing/expected punctuation is attached silently (not a real
        # recovery); only unexpected content tokens count as recoveries.
        _EXPECTED_TRAILING = {"SENT_END", "COMMA", "SEMICOLON", "COLON",
                               "QUOTE", "LPAREN", "RPAREN", "DASH", "ELLIPSIS"}
        while self.i < self.n:
            tok = self.advance()
            if tok.pos not in _EXPECTED_TRAILING:
                self.recoveries += 1
            root.children.append(Node("LEAF", token=tok))
        return root

    def parse_np(self) -> Node:
        node = Node("NP")
        if self.pos_at() == "DET":
            node.children.append(Node("DET", token=self.advance()))
        # adjective phrase (zero or more adjectives)
        adj_tokens = []
        while self.pos_at() == "ADJ":
            adj_tokens.append(self.advance())
        if adj_tokens:
            ap = Node("AP")
            for t in adj_tokens:
                ap.children.append(Node("ADJ", token=t))
            node.children.append(ap)
        # head noun(s)
        head_found = False
        while self.pos_at() in ("NOUN", "PROPN", "PRON", "NUM"):
            node.children.append(Node(self.pos_at(), token=self.advance()))
            head_found = True
            if self.pos_at() == "COMMA" and self.pos_at(1) in ("NOUN", "PROPN"):
                node.children.append(Node("COMMA", token=self.advance()))
        # trailing prepositional phrase(s) -> right recursion, bounded
        while self.pos_at() == "PREP":
            node.children.append(self.parse_pp())
        if not head_found and not node.children:
            # nothing consumed -- force progress to avoid infinite loop
            return None
        return node

    def parse_pp(self) -> Node:
        node = Node("PP")
        node.children.append(Node("PREP", token=self.advance()))
        if self.pos_at() in FIRST["NP"]:
            np = self.parse_np()
            if np:
                node.children.append(np)
        return node

    def parse_vp(self) -> Node:
        node = Node("VP")
        while self.pos_at() in ("MODAL", "AUX"):
            node.children.append(Node(self.pos_at(), token=self.advance()))
        while self.pos_at() == "ADV":
            node.children.append(Node("ADV", token=self.advance()))
        if self.pos_at() == "VERB":
            node.children.append(Node("VERB", token=self.advance()))
        if self.pos_at() in FIRST["NP"]:
            np = self.parse_np()
            if np:
                node.children.append(np)
        while self.pos_at() == "PREP":
            node.children.append(self.parse_pp())
        while self.pos_at() in ("ADV_CONN", "ADV"):
            node.children.append(Node(self.pos_at(), token=self.advance()))
        return node


def parse_sentence_tokens(tokens) -> Node:
    """Entry point: parse a single sentence's token list -> syntax tree."""
    content = [t for t in tokens]  # keep punctuation for completeness
    p = Parser(content)
    tree = p.parse_sentence()
    tree.attrs["_recoveries"] = p.recoveries
    return tree
