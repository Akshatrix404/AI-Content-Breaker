"""
ll1_grammar.py -- FORMAL GRAMMAR VERIFICATION (LL(1) parsing theory)
=====================================================================
parser.py uses a hand-written recursive-descent parser for speed and ease
of panic-mode recovery. This module is a second, fully independent parsing
engine built the classical compiler-theory way:

    1. A pure BNF grammar (no EBNF '*'/'+' sugar) over the same POS-tag
       terminal alphabet the lexer produces.
    2. Textbook FIRST-set and FOLLOW-set computation (fixed-point worklist
       algorithms, exactly as taught for predictive parsing).
    3. An LL(1) parsing table built from those sets, with explicit conflict
       detection (a cell written twice means the grammar is NOT LL(1) at
       that point -- itself a useful, honestly-reported fact).
    4. A stack-based table-driven parser (the standard LL(1) driving
       algorithm, not recursion) that re-parses the same sentence and is
       used to CROSS-VALIDATE the recursive-descent parser: if the two
       independently-implemented parsers disagree about whether a sentence
       is well-formed, that disagreement is itself surfaced as a signal,
       rather than trusted blindly from a single parser implementation.

Grammar (over POS tags only; punctuation is stripped before this parser
runs -- it plays no role in constituency structure):

    S            -> FRONT NP VP STAIL
    FRONT        -> ADV_CONN | eps
    STAIL        -> CCONJ S | SCONJ S | eps
    NP           -> DETOPT APSTAR NOUNSEQ PPSTAR
    DETOPT       -> DET | eps
    APSTAR       -> ADJ APSTAR | eps
    NOUNSEQ      -> NTERM NOUNSEQTAIL
    NTERM        -> NOUN | PROPN | PRON | NUM
    NOUNSEQTAIL  -> NTERM NOUNSEQTAIL | eps
    PPSTAR       -> PP PPSTAR | eps
    PP           -> PREP NP
    VP           -> AUXSTAR ADVSTAR VERB NPOPT PPSTAR2 ADVTAILSTAR
    AUXSTAR      -> MODAL AUXSTAR | AUX AUXSTAR | eps
    ADVSTAR      -> ADV ADVSTAR | eps
    NPOPT        -> NP | eps
    PPSTAR2      -> PP PPSTAR2 | eps
    ADVTAILSTAR  -> ADV_CONN ADVTAILSTAR | ADV ADVTAILSTAR | eps
"""

EPS = "ε"
END = "$"

START = "S"

GRAMMAR = {
    "S":           [("FRONT", "NP", "VP", "STAIL")],
    "FRONT":       [("ADV_CONN",), (EPS,)],
    "STAIL":       [("CCONJ", "S"), ("SCONJ", "S"), (EPS,)],
    "NP":          [("DETOPT", "APSTAR", "NOUNSEQ", "PPSTAR")],
    "DETOPT":      [("DET",), (EPS,)],
    "APSTAR":      [("ADJ", "APSTAR"), (EPS,)],
    "NOUNSEQ":     [("NTERM", "NOUNSEQTAIL")],
    "NTERM":       [("NOUN",), ("PROPN",), ("PRON",), ("NUM",)],
    "NOUNSEQTAIL": [("NTERM", "NOUNSEQTAIL"), (EPS,)],
    "PPSTAR":      [("PP", "PPSTAR"), (EPS,)],
    "PP":          [("PREP", "NP")],
    "VP":          [("AUXSTAR", "ADVSTAR", "VERB", "NPOPT", "PPSTAR2", "ADVTAILSTAR")],
    "AUXSTAR":     [("MODAL", "AUXSTAR"), ("AUX", "AUXSTAR"), (EPS,)],
    "ADVSTAR":     [("ADV", "ADVSTAR"), (EPS,)],
    "NPOPT":       [("NP",), (EPS,)],
    "PPSTAR2":     [("PP", "PPSTAR2"), (EPS,)],
    "ADVTAILSTAR": [("ADV_CONN", "ADVTAILSTAR"), ("ADV", "ADVTAILSTAR"), (EPS,)],
}

NON_TERMINALS = set(GRAMMAR.keys())
TERMINALS = {"DET", "ADJ", "NOUN", "PROPN", "PRON", "NUM", "PREP",
             "MODAL", "AUX", "VERB", "ADV", "ADV_CONN", "CCONJ", "SCONJ"}


def compute_first_sets():
    """Textbook fixed-point FIRST-set computation over GRAMMAR."""
    first = {nt: set() for nt in NON_TERMINALS}
    changed = True
    while changed:
        changed = False
        for nt, prods in GRAMMAR.items():
            for rhs in prods:
                if rhs == (EPS,):
                    if EPS not in first[nt]:
                        first[nt].add(EPS)
                        changed = True
                    continue
                add_eps = True
                for sym in rhs:
                    if sym == EPS:
                        continue
                    sym_first = {sym} if sym in TERMINALS else first[sym]
                    before = len(first[nt])
                    first[nt] |= (sym_first - {EPS})
                    if len(first[nt]) != before:
                        changed = True
                    if EPS not in sym_first:
                        add_eps = False
                        break
                if add_eps and EPS not in first[nt]:
                    first[nt].add(EPS)
                    changed = True
    return first


def first_of_sequence(seq, first):
    """FIRST of a symbol sequence (used while computing FOLLOW sets)."""
    result = set()
    for sym in seq:
        if sym == EPS:
            continue
        sym_first = {sym} if sym in TERMINALS else first[sym]
        result |= (sym_first - {EPS})
        if EPS not in sym_first:
            return result, False
    return result, True  # whole sequence can derive epsilon


def compute_follow_sets(first):
    """Textbook fixed-point FOLLOW-set computation over GRAMMAR."""
    follow = {nt: set() for nt in NON_TERMINALS}
    follow[START].add(END)
    changed = True
    while changed:
        changed = False
        for nt, prods in GRAMMAR.items():
            for rhs in prods:
                if rhs == (EPS,):
                    continue
                for i, sym in enumerate(rhs):
                    if sym not in NON_TERMINALS:
                        continue
                    rest = rhs[i + 1:]
                    rest_first, rest_nullable = first_of_sequence(rest, first)
                    before = len(follow[sym])
                    follow[sym] |= rest_first
                    if rest_nullable:
                        follow[sym] |= follow[nt]
                    if len(follow[sym]) != before:
                        changed = True
    return follow


def build_parsing_table(first, follow):
    """
    Builds the LL(1) parsing table: table[(nonterminal, terminal)] = rhs.
    Returns (table, conflicts) -- conflicts is a list of cells where more
    than one production wanted the same cell, i.e. the grammar is not
    LL(1) there (reported honestly rather than silently overwritten).
    """
    table = {}
    conflicts = []
    for nt, prods in GRAMMAR.items():
        for rhs in prods:
            rhs_first, nullable = first_of_sequence(rhs, first) if rhs != (EPS,) else (set(), True)
            for term in rhs_first:
                key = (nt, term)
                if key in table and table[key] != rhs:
                    conflicts.append({"nonterminal": nt, "terminal": term,
                                       "existing": " ".join(table[key]), "new": " ".join(rhs)})
                table[key] = rhs
            if nullable:
                for term in follow[nt]:
                    key = (nt, term)
                    if key in table and table[key] != rhs:
                        conflicts.append({"nonterminal": nt, "terminal": term,
                                           "existing": " ".join(table[key]), "new": " ".join(rhs)})
                    table[key] = rhs
    return table, conflicts


FIRST_SETS = compute_first_sets()
FOLLOW_SETS = compute_follow_sets(FIRST_SETS)
PARSING_TABLE, TABLE_CONFLICTS = build_parsing_table(FIRST_SETS, FOLLOW_SETS)
IS_LL1 = len(TABLE_CONFLICTS) == 0


class LL1Parser:
    """Stack-based (non-recursive) table-driven LL(1) parser -- the
    standard LL(1) driving algorithm: push START, then repeatedly expand
    the nonterminal on top of the stack using PARSING_TABLE until only
    terminals remain, matching them against the input one at a time.
    Simple error recovery: on a mismatch, skip the offending input symbol
    and retry (panic-mode by terminal-deletion), so one bad token doesn't
    abort the whole trace -- mirroring parser.py's own recovery philosophy,
    but via table lookups instead of hand-written branches.
    """

    def parse(self, pos_sequence):
        tokens = list(pos_sequence) + [END]
        stack = [END, START]
        trace = []
        errors = []
        i = 0
        steps_budget = 2000  # guard against pathological inputs
        while stack and steps_budget > 0:
            steps_budget -= 1
            top = stack.pop()
            cur = tokens[i] if i < len(tokens) else END
            if top == EPS:
                continue
            if top in TERMINALS or top == END:
                if top == cur:
                    i += 1
                else:
                    errors.append({"expected": top, "found": cur, "position": i})
                    if cur != END:
                        stack.append(top)   # retry same expectation on next token
                        i += 1
                    else:
                        break
            else:
                entry = PARSING_TABLE.get((top, cur))
                if entry is None:
                    errors.append({"expected": f"<{top}>", "found": cur, "position": i})
                    if cur != END:
                        stack.append(top)   # retry same nonterminal on next token
                        i += 1
                    else:
                        break               # can't recover at end of input
                else:
                    trace.append(f"{top} -> {' '.join(entry)}")
                    if entry != (EPS,):
                        for sym in reversed(entry):
                            stack.append(sym)
        accepted = (not errors) and i >= len(tokens) - 1
        return {
            "accepted": accepted,
            "production_trace": trace,
            "errors": errors,
            "productions_applied": len(trace),
        }


def grammar_summary():
    """Serializable FIRST/FOLLOW/table-conflict summary for the UI."""
    return {
        "is_ll1": IS_LL1,
        "non_terminal_count": len(NON_TERMINALS),
        "terminal_count": len(TERMINALS),
        "production_count": sum(len(v) for v in GRAMMAR.values()),
        "first_sets": {nt: sorted(s) for nt, s in FIRST_SETS.items()},
        "follow_sets": {nt: sorted(s) for nt, s in FOLLOW_SETS.items()},
        "conflicts": TABLE_CONFLICTS,
        "table_size": len(PARSING_TABLE),
    }
