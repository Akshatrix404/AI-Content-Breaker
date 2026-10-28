"""
cfg_analysis.py -- DISCOURSE CONTROL-FLOW GRAPH & CYCLOMATIC COMPLEXITY
==========================================================================
Builds a control-flow graph the way a compiler builds one for a function's
basic blocks, except the "basic blocks" are sentences: a mandatory
sequential edge i -> i+1 (the document's linear flow), plus an extra
"cohesion" edge i -> j (j > i+1) whenever sentence j reuses a head noun
introduced back in sentence i -- a long-range back-reference, structurally
the same thing a loop-back edge is in a real CFG.

McCabe's cyclomatic complexity M = E - N + 2P (P = number of connected
components, here always 1 for a single document) is then computed over
this graph exactly as a static-analysis tool would compute it for a
function body. A document whose sentences all reference back to earlier
ones densely has a structurally more "tangled" discourse graph -- a higher
M -- than one that reads as a flat, independent sequence.
"""


def _entities(sr):
    return {t["value"].lower().rstrip("s") or t["value"].lower()
            for t in sr["tokens"] if t["pos"] in ("NOUN", "PROPN")}


def build_discourse_cfg(sentence_reports):
    n = len(sentence_reports)
    nodes = list(range(n))
    edges = []
    predecessors = {i: [] for i in range(n)}

    # mandatory sequential flow edges
    for i in range(n - 1):
        edges.append({"src": i, "dst": i + 1, "kind": "sequential"})
        predecessors[i + 1].append(i)

    # cohesion / back-reference edges: sentence j reuses an entity first
    # seen in an earlier, non-adjacent sentence i
    ent_lists = [_entities(sr) for sr in sentence_reports]
    first_seen = {}
    for j in range(n):
        for e in ent_lists[j]:
            if e in first_seen:
                i = first_seen[e]
                if j - i > 1:
                    edges.append({"src": i, "dst": j, "kind": "cohesion", "entity": e})
                    if i not in predecessors[j]:
                        predecessors[j].append(i)
            else:
                first_seen[e] = j

    E = len(edges)
    N = n
    P = 1 if n > 0 else 0
    cyclomatic = E - N + 2 * P if n > 0 else 0

    return {
        "nodes": nodes,
        "edges": edges,
        "edge_count": E,
        "node_count": N,
        "cyclomatic_complexity": cyclomatic,
        "density": round(cyclomatic / n, 3) if n else 0.0,
        "predecessors": predecessors,
    }
