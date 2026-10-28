"""
scorer.py -- CODE GENERATION / REPORT SYNTHESIS PHASE
========================================================
Combines every feature computed across the pipeline -- structural
(attribute-grammar-derived), lexical/discourse, and information-theoretic --
into one explainable weighted report. Every feature is normalised to a
0..1 sub-score, tagged with a category, combined with a fixed documented
weight, and the weighted sum becomes the headline percentage. No black-box
ML step anywhere: every number is traceable back to a specific rule.
"""

import statistics as stats


def _cv(values):
    """Coefficient of variation (stdev / mean) -- 0 means perfectly uniform."""
    if not values or len(values) < 2:
        return None
    mean = stats.mean(values)
    if mean == 0:
        return None
    return stats.pstdev(values) / mean


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


CATEGORIES = {
    "structure": "Syntactic Structure (Attribute Grammar)",
    "lexical": "Lexical & Discourse Patterns",
    "info_theory": "Information-Theoretic",
    "grammar_formal": "Formal Grammar Verification (LL(1))",
    "ir": "Intermediate Representation & Optimization",
    "dataflow": "Data-Flow Analysis (Reaching Definitions)",
    "cfg": "Discourse Control-Flow Graph",
}

# Original 12-feature weights, scaled by 0.80 to make room for the five new
# compiler-theory-derived features below (new features sum to 0.20) while
# keeping the grand total at exactly 1.0.
_BASE_WEIGHTS = {
    "burstiness":                      ("structure",   0.14),
    "depth_uniformity":                 ("structure",   0.07),
    "structural_repetition":             ("structure",   0.13),
    "clause_complexity_uniformity":       ("structure",   0.06),
    "pos_bigram_predictability":           ("structure",   0.10),

    "transition_density":                    ("lexical",     0.12),
    "opener_repetition":                      ("lexical",     0.06),
    "lexical_uniformity":                      ("lexical",     0.05),
    "sentence_similarity_repetition":           ("lexical",     0.08),
    "contraction_scarcity":                      ("lexical",     0.05),

    "punctuation_diversity":                      ("info_theory", 0.04),
    "compression_predictability":                  ("info_theory", 0.10),
}
_SCALE = 0.80
WEIGHTS = {k: (cat, round(w * _SCALE, 6)) for k, (cat, w) in _BASE_WEIGHTS.items()}
WEIGHTS.update({
    "ll1_rejection_rate":        ("grammar_formal", 0.04),
    "ir_redundancy_ratio":       ("ir",              0.05),
    "ir_dead_modifier_ratio":    ("ir",              0.03),
    "unresolved_reference_ratio": ("dataflow",        0.03),
    "discourse_cyclomatic_density": ("cfg",           0.05),
})
assert abs(sum(w for _, w in WEIGHTS.values()) - 1.0) < 1e-6


import json as _json, os as _os

def _load_cal():
    p = _os.path.join(_os.path.dirname(__file__), "data", "calibration.json")
    try:
        return _json.load(open(p))["human_median"]
    except Exception:
        return {}


_CAL = _load_cal()


def _apply_calibration(features):
    """Empirical calibration against a human-written corpus (see calibrate.py):
    a feature's score is measured relative to what genuine human text scores on it.
    calibrated = clip((raw - human_median) / (1 - human_median)); features where humans already
    sit at ~100% carry no signal and are zeroed."""
    if _os.environ.get("AICB_NO_CAL") or not _CAL:
        return
    for k, f in features.items():
        m = _CAL.get(k)
        if m is None:
            continue
        raw = f["score"]
        f["score"] = 0.0 if m >= 0.95 else _clip((raw - m) / (1 - m))
        f["note"] += f"  [calibrated vs human corpus: raw {raw*100:.0f}%, human median {m*100:.0f}%]"


def score_document(symtab, punctuation_counts, num_sentences, clause_depths, ling_scored,
                    ll1_summary=None, ir_summary=None, dataflow_summary=None, cfg_summary=None):
    features = {}

    cv_len = _cv(symtab.sentence_lengths)
    if cv_len is None:
        burst_score, burst_note = 0.5, "Not enough sentences to measure burstiness."
    else:
        burst_score = _clip(1 - cv_len / 0.65)
        burst_note = f"Sentence-length coefficient of variation = {cv_len:.2f} (lower = more uniform)."
    features["burstiness"] = {"score": burst_score, "note": burst_note, "label": "Sentence-Length Burstiness"}

    cv_depth = _cv(symtab.sentence_depths)
    if cv_depth is None:
        depth_score, depth_note = 0.5, "Not enough sentences to measure structural variance."
    else:
        depth_score = _clip(1 - cv_depth / 0.55)
        depth_note = f"Parse-tree depth coefficient of variation = {cv_depth:.2f}."
    features["depth_uniformity"] = {"score": depth_score, "note": depth_note, "label": "Syntax-Tree Shape Uniformity"}

    total_sigs = sum(symtab.rule_sig_freq.values())
    unique_sigs = len(symtab.rule_sig_freq)
    if total_sigs == 0:
        rep_score, rep_note = 0.5, "No production signatures collected."
    else:
        repeated_fraction = 1 - (unique_sigs / total_sigs)
        rep_score = _clip(repeated_fraction / 0.55)
        rep_note = f"{repeated_fraction*100:.0f}% of grammar-rule usages are repeats of an already-seen tree shape."
    features["structural_repetition"] = {"score": rep_score, "note": rep_note, "label": "Grammar-Rule Repetition"}

    cv_clause = _cv(clause_depths)
    if cv_clause is None:
        clause_score, clause_note = 0.5, "Not enough sentences to measure clause-nesting variance."
    else:
        clause_score = _clip(1 - cv_clause / 0.5)
        clause_note = f"Subordinate-clause nesting depth CV = {cv_clause:.2f} (lower = uniformly simple/complex)."
    features["clause_complexity_uniformity"] = {
        "score": clause_score, "note": clause_note, "label": "Clause-Nesting Uniformity"}

    features["pos_bigram_predictability"] = {
        "score": ling_scored["pos_bigram_predictability"]["score"],
        "note": ling_scored["pos_bigram_predictability"]["note"],
        "label": ling_scored["pos_bigram_predictability"]["label"],
    }

    words = max(symtab.total_words, 1)
    per100 = len(symtab.transition_hits) / words * 100
    trans_score = _clip(per100 / 3.5)
    features["transition_density"] = {
        "score": trans_score,
        "note": f"{per100:.2f} discourse connectives (moreover/furthermore/thus/...) per 100 words.",
        "label": "Discourse-Connective Density"}

    repeated_openers = symtab.top_repeated_openers(5)
    if num_sentences <= 1:
        open_score, open_note = 0.0, "Only one sentence -- opener repetition not applicable."
    else:
        max_repeat = max((c for _, c in repeated_openers), default=1)
        open_score = _clip((max_repeat - 1) / max(1, num_sentences * 0.35))
        top = repeated_openers[0] if repeated_openers else None
        open_note = (f"Sentence opener '{top[0]}' reused {top[1]} times." if top
                     else "No repeated sentence openers detected.")
    features["opener_repetition"] = {"score": open_score, "note": open_note, "label": "Repeated Sentence Openers"}

    hapax = symtab.hapax_ratio()
    lex_score = _clip(1 - hapax)
    features["lexical_uniformity"] = {
        "score": lex_score,
        "note": f"{hapax*100:.0f}% of distinct words occur exactly once (hapax legomena).",
        "label": "Lexical Repetition"}

    features["sentence_similarity_repetition"] = {
        "score": ling_scored["sentence_similarity_repetition"]["score"],
        "note": ling_scored["sentence_similarity_repetition"]["note"],
        "label": ling_scored["sentence_similarity_repetition"]["label"],
    }

    features["contraction_scarcity"] = {
        "score": ling_scored["contraction_scarcity"]["score"],
        "note": ling_scored["contraction_scarcity"]["note"],
        "label": ling_scored["contraction_scarcity"]["label"],
    }

    distinct_punct = len([k for k, v in punctuation_counts.items() if v > 0])
    punct_score = _clip(1 - (distinct_punct / 6))
    features["punctuation_diversity"] = {
        "score": punct_score,
        "note": f"{distinct_punct} distinct punctuation types used across the document.",
        "label": "Punctuation Variety (inverse)"}

    features["compression_predictability"] = {
        "score": ling_scored["compression_predictability"]["score"],
        "note": ling_scored["compression_predictability"]["note"],
        "label": ling_scored["compression_predictability"]["label"],
    }

    # ---- New compiler-theory-derived features ---------------------------
    if ll1_summary is not None:
        rej = ll1_summary.get("rejection_ratio", 0.0)
        # Calibration: a human control text was rejected on 100% of sentences, so a high rejection
        # rate reflects the toy grammar's coverage, not authorship. Only the *low*-rejection side
        # (very simple, template-like sentences all accepted) is treated as evidence.
        ll1_score = _clip(rej) if rej < 0.5 else 0.5
        ll1_note = (f"{ll1_summary.get('rejected', 0)} of {ll1_summary.get('total', 0)} sentence(s) "
                    f"rejected by the strict table-driven LL(1) parser (grammar has "
                    f"{ll1_summary.get('conflicts', 0)} formal conflict cell(s)).")
    else:
        ll1_score, ll1_note = 0.5, "LL(1) cross-check not available."
    features["ll1_rejection_rate"] = {"score": ll1_score, "note": ll1_note,
                                       "label": "LL(1) Table-Parser Rejection Rate"}

    if ir_summary is not None:
        ir_red_score = _clip(ir_summary.get("redundancy_ratio", 0.0) / 0.35)
        ir_red_note = (f"{ir_summary.get('cse_eliminated', 0)} of {ir_summary.get('total', 0)} IR instruction(s) "
                       f"were exact repeats of an earlier proposition (common subexpression elimination).")
    else:
        ir_red_score, ir_red_note = 0.5, "IR generation not available."
    features["ir_redundancy_ratio"] = {"score": ir_red_score, "note": ir_red_note,
                                        "label": "Repeated Propositions (IR / CSE)"}

    if ir_summary is not None:
        ir_dead_score = _clip(ir_summary.get("dead_modifier_ratio", 0.0) / 0.25)
        ir_dead_note = (f"{ir_summary.get('dce_removed', 0)} of {ir_summary.get('total', 0)} modifier "
                        f"instruction(s) carried a generic filler word (dead-code elimination pass).")
    else:
        ir_dead_score, ir_dead_note = 0.5, "IR generation not available."
    features["ir_dead_modifier_ratio"] = {"score": ir_dead_score, "note": ir_dead_note,
                                           "label": "Generic Filler Modifiers (IR / DCE)"}

    if dataflow_summary is not None and dataflow_summary.get("total_pronouns", 0) == 0:
        df_score = 0.5
        df_note = ("0 pronoun uses in the text -- no reference to resolve, so this feature carries no "
                   "evidence either way (neutral 50%).")
    elif dataflow_summary is not None:
        df_score = _clip(1 - dataflow_summary.get("unresolved_ratio", 0.0))
        df_note = (f"{len(dataflow_summary.get('unresolved_references', []))} of "
                   f"{dataflow_summary.get('total_pronouns', 0)} pronoun use(s) have no reaching "
                   f"entity definition (reaching-definitions data-flow analysis).")
    else:
        df_score, df_note = 0.5, "Data-flow analysis not available."
    features["unresolved_reference_ratio"] = {"score": df_score, "note": df_note,
                                               "label": "Coreference Resolution (Data-Flow)"}

    if cfg_summary is not None:
        cfg_score = _clip(cfg_summary.get("density", 0.0) / 1.2)
        cfg_note = (f"Discourse control-flow graph: {cfg_summary.get('edge_count', 0)} edges over "
                   f"{cfg_summary.get('node_count', 0)} sentence-nodes -> cyclomatic complexity "
                   f"M = {cfg_summary.get('cyclomatic_complexity', 0)}.")
    else:
        cfg_score, cfg_note = 0.5, "Discourse CFG not available."
    features["discourse_cyclomatic_density"] = {"score": cfg_score, "note": cfg_note,
                                                 "label": "Discourse Cyclomatic Complexity"}

    _apply_calibration(features)
    weighted_total = sum(features[k]["score"] * w for k, (_, w) in WEIGHTS.items())
    percentage = round(weighted_total * 100, 1)

    if percentage >= 24:
        verdict = "Likely AI-Generated"
    elif percentage >= 16:
        verdict = "Mixed / Inconclusive"
    else:
        verdict = "Likely Human-Written"

    confidence = "Low" if symtab.total_words < 80 or num_sentences < 4 else (
        "Medium" if symtab.total_words < 200 else "High")

    breakdown = []
    category_totals = {c: 0.0 for c in CATEGORIES}
    for key, (cat, weight) in WEIGHTS.items():
        f = features[key]
        contribution = f["score"] * weight * 100
        category_totals[cat] += contribution
        breakdown.append({
            "key": key,
            "category": cat,
            "category_label": CATEGORIES[cat],
            "label": f["label"],
            "score_pct": round(f["score"] * 100, 1),
            "weight_pct": round(weight * 100, 1),
            "contribution_pct": round(contribution, 1),
            "note": f["note"],
        })
    breakdown.sort(key=lambda x: -x["contribution_pct"])

    category_breakdown = [
        {"category": c, "label": CATEGORIES[c], "contribution_pct": round(v, 1)}
        for c, v in category_totals.items()
    ]
    category_breakdown.sort(key=lambda x: -x["contribution_pct"])

    return {
        "percentage": percentage,
        "verdict": verdict,
        "confidence": confidence,
        "breakdown": breakdown,
        "category_breakdown": category_breakdown,
    }
