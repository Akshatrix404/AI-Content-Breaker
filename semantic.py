"""
semantic.py -- SEMANTIC ANALYSIS PHASE
=========================================
Compilers run a semantic pass after parsing to flag issues that are
syntactically legal but semantically suspicious (unused variables, type
mismatches...). Here we run an analogous pass over the document's symbol
table + extended linguistic attributes to raise human-readable DIAGNOSTICS,
each tagged with a severity, a category, and (where possible) the sentence
indices it concerns -- exactly like a compiler pointing at a line number.
"""

from dataclasses import dataclass, field


@dataclass
class Diagnostic:
    severity: str      # "info" | "warning" | "flag"
    category: str       # "structure" | "lexical" | "info_theory" | "parser"
    message: str
    sentence_indices: list = field(default_factory=list)


def analyze(symtab, num_sentences, parse_recoveries_total, clause_depths,
            ling_informational, ling_scored, passive_ratio):
    diags = []

    # ---- Structural (attribute-grammar derived) -----------------------------
    top_sigs = symtab.top_repeated_rule_signatures(3)
    for sig, count in top_sigs:
        if count >= max(3, num_sentences * 0.3):
            diags.append(Diagnostic(
                "flag", "structure",
                f"Grammar-rule pattern '{sig}' repeats {count} times across the document "
                f"-- unusually uniform sentence construction.",
            ))

    if len(clause_depths) >= 3:
        import statistics as stats
        mean_c = stats.mean(clause_depths)
        sd_c = stats.pstdev(clause_depths)
        cv_c = sd_c / mean_c if mean_c else 0
        if cv_c < 0.20:
            diags.append(Diagnostic(
                "flag", "structure",
                f"Clause-nesting depth is nearly constant across sentences (avg={mean_c:.1f}, CV={cv_c:.2f}) "
                f"-- human writing usually mixes simple and complex sentences more.",
            ))

    if len(symtab.sentence_lengths) >= 3:
        import statistics as stats
        mean_len = stats.mean(symtab.sentence_lengths)
        sd = stats.pstdev(symtab.sentence_lengths)
        cv = sd / mean_len if mean_len else 0
        if cv < 0.30:
            diags.append(Diagnostic(
                "flag", "structure",
                f"Very low sentence-length variance (CV={cv:.2f}, avg={mean_len:.1f} words) "
                f"-- human writing usually varies pacing more.",
            ))
        elif cv > 0.75:
            diags.append(Diagnostic(
                "info", "structure",
                f"High sentence-length variance (CV={cv:.2f}) -- typical of natural pacing.",
            ))

    pos_note = ling_scored["pos_bigram_predictability"]
    if pos_note["score"] > 0.62:
        diags.append(Diagnostic(
            "flag", "structure",
            f"Syntax stream is highly predictable ({pos_note['note']}).",
        ))

    if passive_ratio is not None and passive_ratio > 0.35:
        diags.append(Diagnostic(
            "warning", "structure",
            f"{passive_ratio*100:.0f}% of verb phrases are passive-voice -- often more common in "
            f"formal/generated text than casual human writing (a soft signal on its own).",
        ))

    # ---- Lexical / discourse -------------------------------------------------
    for word, count in symtab.top_repeated_openers(3):
        if count >= max(2, num_sentences * 0.25):
            diags.append(Diagnostic(
                "warning", "lexical",
                f"Sentence opener '{word.capitalize()}' is reused {count} times.",
            ))

    if symtab.total_words > 0:
        per100 = len(symtab.transition_hits) / symtab.total_words * 100
        if per100 > 3.5:
            sample = ", ".join(sorted({w for _, w in symtab.transition_hits}))[:120]
            diags.append(Diagnostic(
                "flag", "lexical",
                f"High discourse-connective density ({per100:.1f} per 100 words): {sample} ...",
                [i for i, _ in symtab.transition_hits],
            ))

    sim = ling_scored["sentence_similarity_repetition"]
    if sim["score"] > 0.6:
        diags.append(Diagnostic("flag", "lexical", f"Templated phrasing detected: {sim['note']}"))

    contraction = ling_scored["contraction_scarcity"]
    if contraction["score"] > 0.75 and symtab.total_words > 60:
        diags.append(Diagnostic(
            "warning", "lexical",
            f"Almost no contractions used ({contraction['note']}) -- can indicate a formal, "
            f"AI-typical register, though this alone is weak evidence.",
        ))

    # ---- Information-theoretic -------------------------------------------------
    comp = ling_scored["compression_predictability"]
    if comp["score"] > 0.6:
        diags.append(Diagnostic("flag", "info_theory", f"High text redundancy: {comp['note']}"))

    # ---- Parser diagnostics ------------------------------------------------
    if parse_recoveries_total > num_sentences * 2 and num_sentences > 0:
        diags.append(Diagnostic(
            "info", "parser",
            f"Parser needed {parse_recoveries_total} panic-mode recoveries -- input contains "
            f"many fragments or non-standard grammar (could be informal human text, code, or lists).",
        ))

    if not diags:
        diags.append(Diagnostic("info", "structure", "No strong structural anomalies detected."))

    return diags
