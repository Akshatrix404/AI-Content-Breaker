"""
dataflow.py -- DATA-FLOW ANALYSIS (reaching definitions)
============================================================
A direct port of the classic "reaching definitions" data-flow analysis
(the same fixed-point worklist algorithm compilers use to track which
assignments reach which program points) onto discourse: a noun's first
mention is treated as a DEFINITION of that entity; a pronoun later in
the document is a USE that must be reached by some definition, or it is
flagged the same way a compiler flags "use of undefined variable."

For each sentence/node i in the discourse control-flow graph (see
cfg_analysis.py -- this module accepts any predecessor map, defaulting
to the simple linear chain i-1 -> i if none is supplied):

    GEN[i]  = entities first defined in sentence i
    KILL[i] = entities defined in any OTHER sentence (a redefinition
              kills the reach of the prior one -- standard reaching-def
              semantics)
    IN[i]   = union of OUT[p] for every predecessor p
    OUT[i]  = GEN[i] U (IN[i] - KILL[i])

iterated to a fixed point exactly like the textbook worklist algorithm
(terminates in one pass on a simple chain, but the implementation is
general enough for the richer graph cfg_analysis.py builds).
"""


def _entities_in_sentence(sr):
    """Head nouns (lowercased) introduced by NOUN/PROPN tokens in this sentence."""
    ents = []
    for t in sr["tokens"]:
        if t["pos"] in ("NOUN", "PROPN"):
            ents.append(t["value"].lower().rstrip("s") or t["value"].lower())
    return ents


# 1st/2nd-person pronouns are deictic: they point at speaker/reader, so they never need a textual
# antecedent. Calibrated against a human control text, where they were wrongly flagged "unresolved".
DEICTIC = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
           "you", "your", "yours", "yourself"}


def _pronouns_in_sentence(sr):
    return [t["value"].lower() for t in sr["tokens"]
            if t["pos"] == "PRON" and t["value"].lower() not in DEICTIC]


def reaching_definitions(sentence_reports, predecessors=None):
    """
    predecessors: dict sentence_index -> list of predecessor sentence_index.
    Defaults to the simple linear chain (i-1 -> i).
    Returns per-sentence GEN/KILL/IN/OUT sets plus a list of unresolved
    pronoun uses (pronouns with no reaching definition at all).
    """
    n = len(sentence_reports)
    if predecessors is None:
        predecessors = {i: ([i - 1] if i > 0 else []) for i in range(n)}

    gen = [set(_entities_in_sentence(sr)) for sr in sentence_reports]
    # KILL[i] = any entity defined elsewhere (globally) other than in i itself --
    # a later redefinition of the same entity kills the earlier one's reach.
    all_defs = {}
    for i, g in enumerate(gen):
        for e in g:
            all_defs.setdefault(e, []).append(i)
    kill = []
    for i, g in enumerate(gen):
        k = set()
        for e in g:
            if len(all_defs.get(e, [])) > 1:
                k.add(e)
        kill.append(k)

    IN = [set() for _ in range(n)]
    OUT = [set() for _ in range(n)]
    worklist = list(range(n))
    iterations = 0
    while worklist:
        iterations += 1
        i = worklist.pop(0)
        new_in = set()
        for p in predecessors.get(i, []):
            new_in |= OUT[p]
        new_out = gen[i] | (new_in - kill[i])
        changed = (new_in != IN[i]) or (new_out != OUT[i])
        IN[i], OUT[i] = new_in, new_out
        if changed:
            for j in range(n):
                if i in predecessors.get(j, []) and j not in worklist:
                    worklist.append(j)
        if iterations > n * n + 10:
            break  # safety valve

    unresolved = []
    total_pronouns = 0
    for i, sr in enumerate(sentence_reports):
        prons = _pronouns_in_sentence(sr)
        total_pronouns += len(prons)
        if prons and not IN[i] and not gen[i]:
            for pr in prons:
                unresolved.append({"sentence_index": i, "pronoun": pr,
                                    "reason": "no entity definition reaches this sentence"})

    per_sentence = [{
        "index": i,
        "gen": sorted(gen[i]),
        "kill": sorted(kill[i]),
        "in": sorted(IN[i]),
        "out": sorted(OUT[i]),
    } for i in range(n)]

    return {
        "per_sentence": per_sentence,
        "iterations": iterations,
        "unresolved_references": unresolved,
        "total_pronouns": total_pronouns,
        "unresolved_ratio": round(len(unresolved) / total_pronouns, 3) if total_pronouns else 0.0,
    }
