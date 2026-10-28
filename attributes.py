"""
attributes.py -- ATTRIBUTE GRAMMAR EVALUATION PHASE
=====================================================
This is the heart of the "attribute grammar" framework. Every node in the
parse tree produced by parser.py carries a set of ATTRIBUTES computed by
semantic rules attached to each grammar production -- exactly the S-attributed
/ L-attributed evaluation scheme taught in compiler-design courses:

  SYNTHESIZED attributes (computed bottom-up, child -> parent), e.g.:
      node.depth            = 1 + max(child.depth)              [height rule]
      node.width             = len(children)                     [branching rule]
      node.leaf_count         = sum(child.leaf_count)              [count rule]
      node.token_count        = sum(child.token_count)             [count rule]
      node.rule_sig           = "SYMBOL(child_symbols...)"         [structural signature]
      node.subtree_sig        = recursive signature string          [shape fingerprint]

  INHERITED attributes (computed top-down, parent -> child), e.g.:
      child.depth_from_root   = parent.depth_from_root + 1
      child.sentence_index    = parent.sentence_index (propagated context)

The evaluator performs two tree walks (post-order then pre-order), which is
the standard two-pass algorithm for evaluating attribute grammars whose
dependency graph is acyclic but mixes both attribute flows.
"""

from parser import Node


def evaluate_synthesized(node: Node) -> None:
    """Post-order walk: bottom-up attribute computation."""
    if node is None:
        return
    for child in node.children:
        evaluate_synthesized(child)

    if node.is_leaf():
        node.attrs["depth"] = 1
        node.attrs["width"] = 0
        node.attrs["leaf_count"] = 1
        node.attrs["token_count"] = 1
        node.attrs["subtree_sig"] = node.symbol
    else:
        child_depths = [c.attrs.get("depth", 1) for c in node.children]
        node.attrs["depth"] = 1 + (max(child_depths) if child_depths else 0)
        node.attrs["width"] = len(node.children)
        node.attrs["leaf_count"] = sum(c.attrs.get("leaf_count", 0) for c in node.children)
        node.attrs["token_count"] = sum(c.attrs.get("token_count", 0) for c in node.children)
        node.attrs["rule_sig"] = f"{node.symbol}({','.join(c.symbol for c in node.children)})"
        node.attrs["subtree_sig"] = f"{node.symbol}[{'|'.join(c.attrs.get('subtree_sig','?') for c in node.children)}]"


def evaluate_inherited(node: Node, depth_from_root=0, sentence_index=0) -> None:
    """Pre-order walk: top-down attribute propagation."""
    if node is None:
        return
    node.attrs["depth_from_root"] = depth_from_root
    node.attrs["sentence_index"] = sentence_index
    for pos, child in enumerate(node.children):
        child.attrs["sibling_position"] = pos   # inherited context attribute
        evaluate_inherited(child, depth_from_root + 1, sentence_index)


def evaluate_tree(node: Node, sentence_index: int) -> Node:
    """Run both passes of the attribute grammar evaluator on one parse tree."""
    if node is None:
        return node
    evaluate_synthesized(node)
    evaluate_inherited(node, 0, sentence_index)
    return node


def collect_rule_signatures(node: Node, out: list) -> None:
    """Gather every internal production's rule signature used in this tree
    (feeds the document symbol table's structural-repetition counter)."""
    if node is None or node.is_leaf():
        return
    sig = node.attrs.get("rule_sig")
    if sig:
        out.append(sig)
    for c in node.children:
        collect_rule_signatures(c, out)


# ---------------------------------------------------------------------------
# EXTENDED ATTRIBUTES: voice (passive/active) and clause-nesting depth.
# These are additional synthesized attributes computed in dedicated passes,
# demonstrating that an attribute grammar's rule set is open for extension --
# new semantic rules can be layered on the same tree without touching the
# grammar or the parser.
# ---------------------------------------------------------------------------

_IRREGULAR_PARTICIPLES = {
    "been", "done", "made", "given", "taken", "written", "seen", "known",
    "shown", "spoken", "broken", "chosen", "driven", "eaten", "forgotten",
    "gotten", "got", "hidden", "held", "kept", "left", "lost", "meant",
    "paid", "read", "said", "sold", "sent", "set", "told", "thought",
    "understood", "won", "brought", "bought", "built", "caught", "felt",
    "found", "heard", "led", "lent", "met", "put", "run", "sat", "stood",
    "taught", "torn", "woken", "begun", "chosen", "drawn", "fallen",
    "grown", "worn", "sung", "sunk", "swum", "flown",
}


def _looks_like_past_participle(word: str) -> bool:
    lw = word.lower()
    return lw.endswith("ed") or lw in _IRREGULAR_PARTICIPLES


def annotate_voice(node: Node) -> None:
    """Mark every VP node with attrs['voice'] = 'passive' | 'active', using
    the pattern AUX(be-form) ... VERB(past-participle) with no other finite
    verb in between -- a standard heuristic for passive-voice detection."""
    if node is None:
        return
    if node.symbol == "VP":
        has_be_aux = any(
            c.symbol == "AUX" and c.token is not None and
            c.token.value.lower() in {"be", "am", "is", "are", "was", "were", "been", "being"}
            for c in node.children
        )
        verb_child = next((c for c in node.children if c.symbol == "VERB"), None)
        if has_be_aux and verb_child is not None and _looks_like_past_participle(verb_child.token.value):
            node.attrs["voice"] = "passive"
        else:
            node.attrs["voice"] = "active"
    for c in node.children:
        annotate_voice(c)


def compute_clause_depth(node: Node) -> int:
    """Synthesized attribute: how many levels of subordinate/coordinate
    clauses are nested under this S node. depth 1 = a single simple clause."""
    if node is None:
        return 0
    if node.symbol != "S":
        for c in node.children:
            compute_clause_depth(c)
        return 0
    nested = [compute_clause_depth(c) for c in node.children if c.symbol == "S"]
    for c in node.children:
        if c.symbol != "S":
            compute_clause_depth(c)
    depth = 1 + (max(nested) if nested else 0)
    node.attrs["clause_depth"] = depth
    return depth


def collect_passive_flags(node: Node, out: list) -> None:
    if node is None:
        return
    if node.symbol == "VP" and "voice" in node.attrs:
        out.append(node.attrs["voice"] == "passive")
    for c in node.children:
        collect_passive_flags(c, out)
