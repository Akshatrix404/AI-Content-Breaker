"""
pipeline.py -- COMPILATION DRIVER
====================================
Wires together every phase, in the exact order a real compiler front-end
would run them:

    source text
        --(lexer.tokenize)-->                 token stream
        --(lexer.tag_tokens)-->                POS-tagged token stream
        --(lexer.split_sentences)-->            per-sentence token lists
        --(parser.parse_sentence_tokens)-->      parse tree (per sentence)
        --(attributes.evaluate_tree)-->           attribute-annotated parse tree
        --(attributes.annotate_voice / clause)-->  extended attributes (voice, clause depth)
        --(symbol_table.SymbolTable)-->            document symbol table
        --(linguistics.analyze_linguistics)-->      extended lexical/statistical pass
        --(semantic.analyze)-->                      diagnostics
        --(scorer.score_document)-->                  final explainable report
        --(per-sentence suspicion scoring)-->          inline highlight data
"""

import time
import statistics as stats
from collections import Counter

from lexer import tokenize, tag_tokens, split_sentences, TRANSITION_PHRASES_START
from parser import parse_sentence_tokens, Node
from attributes import (evaluate_tree, collect_rule_signatures, annotate_voice,
                          compute_clause_depth, collect_passive_flags)
from symbol_table import SymbolTable
from linguistics import analyze_linguistics, STOPWORDS, HEDGE_WORDS, CONTRACTION_PATTERN, count_syllables
from semantic import analyze as semantic_analyze
from scorer import score_document
from humanizer import humanize
from lexer import TRANSITION_WORDS
from ll1_grammar import LL1Parser, grammar_summary
from ir_gen import extract_sentence_ir, optimize_ir
from dataflow import reaching_definitions
from cfg_analysis import build_discourse_cfg


def _tree_to_dict(node: Node, max_depth=6, depth=0):
    """Serialize a parse tree (with its attributes) to plain dict/JSON for the UI."""
    if node is None:
        return None
    label = node.symbol
    if node.token is not None:
        label = f"{node.symbol}:{node.token.value}"
    d = {
        "symbol": node.symbol,
        "label": label,
        "attrs": {
            "depth": node.attrs.get("depth"),
            "width": node.attrs.get("width"),
            "leaf_count": node.attrs.get("leaf_count"),
            "depth_from_root": node.attrs.get("depth_from_root"),
            "voice": node.attrs.get("voice"),
            "clause_depth": node.attrs.get("clause_depth"),
        },
    }
    if depth < max_depth:
        d["children"] = [_tree_to_dict(c, max_depth, depth + 1) for c in node.children]
    else:
        d["children"] = []
    return d


# ---- Token-level AI-vs-Human "jar" classification ---------------------------
# A second, fully deterministic rule, independent of (but anchored to) the
# document-level scorer above, used to sort each individual WORD token into
# one of two jars. Every token starts from the document's own measured
# AI-percentage (so the jar split reflects the real score, not a guess),
# is nudged a little by its parent sentence's suspicion score, and nudged
# further by a small set of documented token-level signals. Every weight
# below is fixed and printed back to the UI with the exact arithmetic, so
# nothing is a black box.
AI_THRESHOLD = 0.24   # must match the 'Likely AI' threshold in scorer.py
TOKEN_JAR_WEIGHTS = {
    "document_score": 1.00,         # anchor: the document's own measured AI-percentage
    "sentence_suspicion": 0.15,     # small nudge toward the parent sentence's own suspicion score
    "transition_word": 0.18,        # formal discourse connectives ("moreover", "furthermore") -> AI
    "repeated_function_word": 0.08,  # a stopword reused often doc-wide -> AI (uniform, templated diction)
    "hapax_rarity": -0.12,          # a word used only once in the whole document -> Human (specific/idiosyncratic)
    "contraction": -0.15,           # contractions ("don't", "it's") -> Human (informal register)
    "hedge_word": -0.08,            # hedges ("maybe", "I think", "sort of") -> Human (uncertainty AI avoids)
}


def classify_token_jar(tok, sentence_suspicion, document_score):
    """Returns (ai_score 0..1, jar 'ai'|'human', formula: list of applied terms)."""
    terms = []
    # Calibrated scorer: a document at the 24% "Likely AI" threshold must sit exactly on the jar
    # cut-off (0.5), so the anchor is re-centred by +0.26 (0.50 - 0.24).
    anchor = min(1.0, document_score + (0.5 - AI_THRESHOLD))
    score = anchor
    terms.append({"factor": "document_score",
                  "detail": f"document AI score {document_score*100:.1f}% re-centred on the {AI_THRESHOLD*100:.0f}% AI threshold -> {anchor*100:.1f}%",
                  "weight": TOKEN_JAR_WEIGHTS["document_score"], "contribution": round(anchor, 3)})

    c = TOKEN_JAR_WEIGHTS["sentence_suspicion"] * (sentence_suspicion - 0.5)
    score += c
    terms.append({"factor": "sentence_suspicion", "detail": f"parent sentence suspicion = {sentence_suspicion:.2f}",
                  "weight": TOKEN_JAR_WEIGHTS["sentence_suspicion"], "contribution": round(c, 3)})

    if tok.get("is_transition_word"):
        c = TOKEN_JAR_WEIGHTS["transition_word"]
        score += c
        terms.append({"factor": "transition_word", "detail": "formal discourse connective",
                      "weight": c, "contribution": round(c, 3)})

    if tok.get("is_stopword") and (tok.get("doc_frequency") or 0) >= 3:
        c = TOKEN_JAR_WEIGHTS["repeated_function_word"]
        score += c
        terms.append({"factor": "repeated_function_word",
                      "detail": f"function word repeated {tok.get('doc_frequency')}x in document",
                      "weight": c, "contribution": round(c, 3)})

    if tok.get("is_hapax"):
        c = TOKEN_JAR_WEIGHTS["hapax_rarity"]
        score += c
        terms.append({"factor": "hapax_rarity", "detail": "occurs only once in the whole document",
                      "weight": c, "contribution": round(c, 3)})

    if tok.get("is_contraction"):
        c = TOKEN_JAR_WEIGHTS["contraction"]
        score += c
        terms.append({"factor": "contraction", "detail": "informal contraction",
                      "weight": c, "contribution": round(c, 3)})

    if tok.get("is_hedge_word"):
        c = TOKEN_JAR_WEIGHTS["hedge_word"]
        score += c
        terms.append({"factor": "hedge_word", "detail": "hedging / uncertainty marker",
                      "weight": c, "contribution": round(c, 3)})

    score = max(0.0, min(1.0, score))
    jar = "ai" if score >= 0.5 else "human"
    return round(score, 3), jar, terms


def _reconstruct_sentence_text(sent_tokens):
    return "".join(
        (" " if (t.type == "WORD" and i > 0 and sent_tokens[i-1].type == "WORD") else
         ("" if t.type in ("SENT_END", "COMMA", "SEMICOLON", "COLON") else " ")) + t.value
        for i, t in enumerate(sent_tokens)
    ).strip()


def analyze_text(text: str) -> dict:
    t0 = time.perf_counter()
    stages = []

    def stamp(name, detail=""):
        stages.append({"stage": name, "detail": detail,
                        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2)})

    text = (text or "").strip()
    if not text:
        return {"error": "empty_input"}

    # ---- PHASE 1: Lexical analysis -----------------------------------------
    tokens = tokenize(text)
    tag_tokens(tokens)
    sentences_tokens = split_sentences(tokens)
    stamp("Lexical Analysis (tokenize + POS tag)")

    punctuation_counts = Counter()
    for t in tokens:
        if t.type in ("SENT_END", "COMMA", "SEMICOLON", "COLON", "QUOTE",
                       "LPAREN", "RPAREN", "DASH", "ELLIPSIS"):
            punctuation_counts[t.type] += 1

    # ---- PHASE 2 + 3: Syntax analysis + Attribute grammar evaluation ------
    symtab = SymbolTable()
    sentence_reports = []
    total_recoveries = 0
    clause_depths = []
    all_passive_flags = []
    pos_tag_sequence = []
    word_tokens_per_sentence = []
    sentence_texts = []
    sentence_word_sets = []

    for idx, sent_tokens in enumerate(sentences_tokens):
        tree = parse_sentence_tokens(sent_tokens)
        evaluate_tree(tree, idx)
        annotate_voice(tree)
        compute_clause_depth(tree)
        total_recoveries += tree.attrs.get("_recoveries", 0)

        passive_flags = []
        collect_passive_flags(tree, passive_flags)
        all_passive_flags.extend(passive_flags)
        sentence_passive_ratio = (sum(passive_flags) / len(passive_flags)) if passive_flags else 0.0

        word_tokens = [t for t in sent_tokens if t.type == "WORD"]
        word_tokens_per_sentence.append(word_tokens)
        opener = word_tokens[0].value if word_tokens else None
        pos_tag_sequence.extend(t.pos for t in sent_tokens if t.pos)

        rule_sigs = []
        collect_rule_signatures(tree, rule_sigs)

        # LL(1) table-driven cross-check (independent 2nd parser; formal grammar theory)
        ll1_result = LL1Parser().parse([t.pos for t in word_tokens if t.pos])

        # Triple-based IR extraction for this sentence (optimized later, document-wide)
        sentence_ir = extract_sentence_ir(tree, idx)

        symtab.add_sentence(idx, word_tokens, opener, rule_sigs, tree.attrs.get("depth", 1))
        clause_depths.append(tree.attrs.get("clause_depth", 1))

        sentence_text = _reconstruct_sentence_text(sent_tokens)
        sentence_texts.append(sentence_text)
        sentence_word_sets.append({t.value.lower() for t in word_tokens})

        low = sentence_text.lower()
        opens_with_transition = any(low.startswith(p) for p in TRANSITION_PHRASES_START)

        sentence_reports.append({
            "index": idx,
            "text": sentence_text,
            "word_count": len(word_tokens),
            "tree_depth": tree.attrs.get("depth"),
            "clause_depth": tree.attrs.get("clause_depth"),
            "passive_ratio": round(sentence_passive_ratio, 2),
            "recoveries": tree.attrs.get("_recoveries", 0),
            "opens_with_transition_phrase": opens_with_transition,
            "rule_signatures": rule_sigs,
            "tokens": [{"value": t.value, "pos": t.pos} for t in sent_tokens],
            "tree": _tree_to_dict(tree),
            "ll1": ll1_result,
            "ir": sentence_ir,
        })
    stamp("Syntax Analysis + Attribute Grammar Evaluation")
    stamp("Symbol Table Construction")

    # ---- PHASE 3.1: Formal Grammar Verification (LL(1) cross-check) --------
    _g = grammar_summary()
    _rejected = sum(1 for sr in sentence_reports if not sr["ll1"]["accepted"])
    ll1_summary = {
        "total": len(sentence_reports),
        "rejected": _rejected,
        "rejection_ratio": round(_rejected / len(sentence_reports), 3) if sentence_reports else 0.0,
        "conflicts": len(_g["conflicts"]),
        "is_ll1": _g["is_ll1"],
        "grammar": _g,
    }
    stamp("Formal Grammar Verification (LL(1) table-driven cross-check)",
          f"{_rejected}/{len(sentence_reports)} sentence(s) rejected by the strict table parser; "
          f"grammar has {_g['table_size']} table cells, {len(_g['conflicts'])} conflict(s)")

    # ---- PHASE 3.2: Intermediate Representation + Optimization passes ------
    doc_ir_instructions = [instr for sr in sentence_reports for instr in sr["ir"]]
    ir_summary = optimize_ir(doc_ir_instructions)
    stamp("Intermediate Representation Generation + Optimization (CSE / DCE)",
          f"{ir_summary['total']} IR instruction(s) generated, "
          f"{ir_summary['cse_eliminated']} eliminated by CSE, {ir_summary['dce_removed']} by DCE")

    # ---- PHASE 3.3: Discourse Control-Flow Graph + Cyclomatic Complexity ---
    cfg_summary = build_discourse_cfg(sentence_reports)
    stamp("Discourse Control-Flow Graph Construction",
          f"{cfg_summary['node_count']} node(s), {cfg_summary['edge_count']} edge(s) -> "
          f"cyclomatic complexity M = {cfg_summary['cyclomatic_complexity']}")

    # ---- PHASE 3.35: Data-Flow Analysis (reaching definitions) -------------
    dataflow_summary = reaching_definitions(sentence_reports, predecessors=cfg_summary["predecessors"])
    stamp("Data-Flow Analysis (reaching-definitions worklist algorithm)",
          f"fixed point reached in {dataflow_summary['iterations']} iteration(s); "
          f"{len(dataflow_summary['unresolved_references'])}/{dataflow_summary['total_pronouns']} "
          f"pronoun use(s) unresolved")

    # ---- PHASE 3.4: Token-level transparency enrichment --------------------
    # A second pass, now that document-wide frequencies are fully known, adds
    # per-token analysis flags -- this is the "disassembly listing" of the
    # pipeline: every single token, annotated with every rule that touched it.
    for sr in sentence_reports:
        for tok in sr["tokens"]:
            if tok["pos"] not in ("SENT_END", "COMMA", "SEMICOLON", "COLON",
                                    "QUOTE", "LPAREN", "RPAREN", "DASH", "ELLIPSIS", "NUM"):
                lw = tok["value"].lower()
                tok["is_stopword"] = lw in STOPWORDS
                tok["is_transition_word"] = lw in TRANSITION_WORDS
                tok["is_hedge_word"] = lw in HEDGE_WORDS
                tok["is_contraction"] = bool(CONTRACTION_PATTERN.fullmatch(lw))
                freq = symtab.word_freq.get(lw, 0)
                tok["doc_frequency"] = freq
                tok["is_hapax"] = freq == 1
                tok["syllables"] = count_syllables(tok["value"]) if tok["pos"] not in ("PUNCT",) else None
            else:
                tok["is_stopword"] = False
                tok["is_transition_word"] = False
                tok["is_hedge_word"] = False
                tok["is_contraction"] = False
                tok["doc_frequency"] = None
                tok["is_hapax"] = None
                tok["syllables"] = None

    # ---- PHASE 3.5: Extended lexical & statistical analysis ---------------
    ling = analyze_linguistics(text, word_tokens_per_sentence, pos_tag_sequence,
                                 punctuation_counts, sentence_texts)
    passive_ratio_doc = (sum(all_passive_flags) / len(all_passive_flags)) if all_passive_flags else None
    stamp("Extended Lexical & Statistical Analysis")

    # ---- PHASE 4: Semantic analysis (diagnostics) --------------------------
    diagnostics = semantic_analyze(symtab, len(sentences_tokens), total_recoveries,
                                     clause_depths, ling["informational"], ling["scored"],
                                     passive_ratio_doc)
    stamp("Semantic Analysis (diagnostics)")

    # ---- PHASE 5: Code generation / final report ---------------------------
    report = score_document(symtab, punctuation_counts, len(sentences_tokens),
                              clause_depths, ling["scored"],
                              ll1_summary=ll1_summary, ir_summary=ir_summary,
                              dataflow_summary=dataflow_summary, cfg_summary=cfg_summary)
    stamp("Report Synthesis (scoring)")

    # ---- PHASE 6: Per-sentence suspicion scoring (for inline highlighting) -
    mean_wc = stats.mean(symtab.sentence_lengths) if symtab.sentence_lengths else 0
    sd_wc = stats.pstdev(symtab.sentence_lengths) if len(symtab.sentence_lengths) > 1 else 0
    sd_wc = sd_wc or (mean_wc * 0.3 if mean_wc else 1) or 1
    rule_sig_freq = symtab.rule_sig_freq

    for i, sr in enumerate(sentence_reports):
        suspicion = 0.0
        notes = []
        z = abs(sr["word_count"] - mean_wc) / sd_wc if sd_wc else 0
        length_contrib = 0.28 * max(0, 1 - z)
        suspicion += length_contrib
        if length_contrib > 0.16:
            notes.append(f"length ({sr['word_count']}w) sits close to the document average ({mean_wc:.1f}w)")
        if sr["opens_with_transition_phrase"]:
            suspicion += 0.24
            notes.append("opens with a formal transition phrase")
        repeats = sum(1 for sig in sr["rule_signatures"] if rule_sig_freq.get(sig, 0) > 1)
        has_repeated_structure = False
        if sr["rule_signatures"]:
            frac = repeats / len(sr["rule_signatures"])
            suspicion += 0.24 * min(1.0, frac)
            if frac > 0.4:
                has_repeated_structure = True
                notes.append("built from grammar-rule shapes that repeat elsewhere in the document")
        if i > 0 and sentence_word_sets[i - 1] and sentence_word_sets[i]:
            wa, wb = sentence_word_sets[i - 1], sentence_word_sets[i]
            jaccard = len(wa & wb) / len(wa | wb) if (wa | wb) else 0
            sim_contrib = 0.24 * min(1.0, jaccard / 0.35)
            suspicion += sim_contrib
            if sim_contrib > 0.14:
                notes.append("shares notable word overlap with the previous sentence")
        if sr["passive_ratio"] and sr["passive_ratio"] > 0.5:
            notes.append("contains passive-voice construction")
        if sr["recoveries"] > 2:
            notes.append(f"parser needed {sr['recoveries']} recovery steps (unusual grammar)")
        sr["suspicion"] = round(min(1.0, suspicion), 2)
        sr["has_repeated_structure"] = has_repeated_structure
        sr["notes"] = notes
        del sr["rule_signatures"]

    # ---- PHASE 6.5: Token jar classification (AI jar / Human jar) ----------
    jar_tokens = []
    ai_count = 0
    human_count = 0
    for sr in sentence_reports:
        for tok in sr["tokens"]:
            if tok["pos"] in ("SENT_END", "COMMA", "SEMICOLON", "COLON",
                               "QUOTE", "LPAREN", "RPAREN", "DASH", "ELLIPSIS", "NUM"):
                tok["ai_score"] = None
                tok["jar"] = None
                tok["jar_formula"] = None
                continue
            ai_score, jar, terms = classify_token_jar(tok, sr["suspicion"], report["percentage"] / 100)
            tok["ai_score"] = ai_score
            tok["jar"] = jar
            tok["jar_formula"] = terms
            if jar == "ai":
                ai_count += 1
            else:
                human_count += 1
            jar_tokens.append({"value": tok["value"], "sentence_index": sr["index"],
                               "ai_score": ai_score, "jar": jar})

    total_classified = ai_count + human_count
    token_jars = {
        "ai_count": ai_count,
        "human_count": human_count,
        "total_classified": total_classified,
        "ai_jar_pct": round(ai_count / total_classified * 100, 1) if total_classified else 0,
        "human_jar_pct": round(human_count / total_classified * 100, 1) if total_classified else 0,
        "tokens": jar_tokens,
        "weights": TOKEN_JAR_WEIGHTS,
    }

    distinct_tags = len(set(t.pos for t in tokens if t.pos))
    stages[0]["detail"] = f"{len(tokens)} tokens produced, {distinct_tags} distinct POS categories used"
    stages[1]["detail"] = (f"{len(sentences_tokens)} parse tree(s) built, "
                            f"{total_recoveries} panic-mode recovery step(s) across all sentences")
    stages[2]["detail"] = (f"{len(symtab.word_freq)} distinct words, "
                            f"{len(symtab.rule_sig_freq)} distinct grammar-rule signatures tracked")
    # indices 3-6 (LL(1) / IR / CFG / data-flow) already carry their own
    # detail strings set at stamp() time above.
    stages[7]["detail"] = (f"POS-bigram entropy {ling['informational']['pos_bigram_entropy_bits']} bits, "
                            f"compression ratio "
                            f"{ling['informational']['compression_ratio_pct']}%"
                            if ling['informational']['compression_ratio_pct'] is not None else
                            f"POS-bigram entropy {ling['informational']['pos_bigram_entropy_bits']} bits")
    stages[8]["detail"] = f"{len(diagnostics)} diagnostic(s) raised"
    stages[9]["detail"] = f"final score {report['percentage']}% -> {report['verdict']}"

    return {
        "input_stats": {
            "characters": len(text),
            "words": symtab.total_words,
            "sentences": len(sentences_tokens),
        },
        "pipeline_stages": stages,
        "symbol_table": symtab.to_dict(),
        "linguistics": ling["informational"],
        "diagnostics": [{"severity": d.severity, "category": d.category, "message": d.message,
                          "sentence_indices": d.sentence_indices} for d in diagnostics],
        "sentences": sentence_reports,
        "token_jars": token_jars,
        "report": report,
        "parser_recoveries_total": total_recoveries,
        "passive_ratio": round(passive_ratio_doc, 2) if passive_ratio_doc is not None else None,
        "ll1_analysis": ll1_summary,
        "ir_analysis": ir_summary,
        "dataflow_analysis": dataflow_summary,
        "cfg_analysis": cfg_summary,
    }


def humanize_and_compare(text: str, deep: bool = False) -> dict:
    """
    Runs the full analysis on the original text, applies the deterministic
    rule-based humanizer passes, then re-runs the exact same analysis on the
    rewritten text -- so the "improvement" isn't asserted, it's measured with
    the same transparent scorer used everywhere else in the app.
    """
    original_report = analyze_text(text)
    if "error" in original_report:
        return original_report

    opener_freq = {w: c for w, c in original_report["symbol_table"]["repeated_openers"]}
    result = humanize(text, opener_freq=opener_freq,
                      score_fn=lambda t: analyze_text(t)["report"]["percentage"] if "error" not in analyze_text(t) else 100.0)
    humanized_text = result["humanized_text"]
    deep_log, deep_error, rag_hits = None, None, None
    if deep:
        try:
            from deep_humanizer import deep_humanize
            sc = lambda t: analyze_text(t)["report"]["percentage"] if "error" not in analyze_text(t) else 100.0
            bd = lambda t: analyze_text(t)["report"]["breakdown"]
            humanized_text, _, deep_log, rag_hits = deep_humanize(text, sc, bd, polish_fn=lambda t: humanize(t, score_fn=sc)['humanized_text'], baseline=humanized_text)
        except Exception as e:  # no key / network: keep the rule-based result
            deep_error = str(e)

    humanized_report = analyze_text(humanized_text) if humanized_text.strip() else None

    return {
        "original_text": text,
        "humanized_text": humanized_text,
        "original_score": original_report["report"]["percentage"],
        "original_verdict": original_report["report"]["verdict"],
        "humanized_score": humanized_report["report"]["percentage"] if humanized_report else None,
        "humanized_verdict": humanized_report["report"]["verdict"] if humanized_report else None,
        "score_delta": round(original_report["report"]["percentage"] -
                              humanized_report["report"]["percentage"], 1) if humanized_report else None,
        "edits": result["edits"],
        "passes_summary": result["passes_summary"],
        "total_edits": result["total_edits"],
        "original_breakdown": original_report["report"]["breakdown"],
        "humanized_breakdown": humanized_report["report"]["breakdown"] if humanized_report else None,
        "deep_log": deep_log, "rag_hits": [{"sim": s, "text": d} for s, d in (rag_hits or [])], "deep_error": deep_error,
    }


def analyze_batch(text: str) -> dict:
    """Split input into paragraphs (blank-line separated) and analyze each
    independently -- useful for spotting a single AI-inserted paragraph
    inside an otherwise human-written document."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    results = []
    for i, para in enumerate(paragraphs):
        r = analyze_text(para)
        if "error" in r:
            continue
        results.append({
            "index": i,
            "preview": para[:140] + ("…" if len(para) > 140 else ""),
            "percentage": r["report"]["percentage"],
            "verdict": r["report"]["verdict"],
            "confidence": r["report"]["confidence"],
            "word_count": r["input_stats"]["words"],
        })
    overall = round(stats.mean([r["percentage"] for r in results]), 1) if results else 0
    return {"paragraphs": results, "overall_average": overall, "count": len(results)}
