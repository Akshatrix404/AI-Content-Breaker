# AI Content Breaker
### An Attribute-Grammar-Based Framework for AI-Generated Text Detection

A fully working, explainable text-analysis system built entirely on classic
**compiler-design theory** — no black-box ML model, no external API calls.
Every phase you'd find in a real compiler front end is implemented and
applied to natural-language text instead of source code, and every number
in the final report can be traced back to a specific grammar rule or formula.

---

## 1. Why compiler theory for AI-text detection?

Large-language-model output tends to be *structurally* more regular than
human writing: sentence lengths cluster tightly, the same syntactic "shapes"
recur, clause complexity stays flat, and a small set of discourse
connectives (*moreover, furthermore, additionally, in conclusion...*) gets
reused heavily. A compiler front end is exactly the machinery designed to
notice structural regularities — so this project repurposes it, and adds a
second, information-theoretic layer (entropy, compression, similarity) on top.

| Compiler phase | Real compilers | This project |
|---|---|---|
| Lexical analysis | tokenize source into keywords/identifiers | tokenize text, POS-tag every word |
| Syntax analysis | parse tokens into an AST via a CFG | recursive-descent parser builds a sentence parse tree from a hand-written English CFG (POS tags as terminals), with panic-mode error recovery |
| Attribute grammar evaluation | compute synthesized/inherited attributes over the AST | depth, width, leaf/token counts, rule-signatures, **voice (passive/active)**, **clause-nesting depth** |
| Symbol table | track identifiers, scope, usage counts | word/bigram frequency, repeated tree-shapes, repeated sentence openers |
| Extended lexical/statistical pass *(new)* | peephole/stats passes in real compilers | readability, POS-bigram entropy, DEFLATE compression ratio, sentence-similarity, contraction/pronoun/hedge analysis |
| Semantic analysis | flag type errors, unused variables, etc. | 12+ categorized diagnostics (structure / lexical / info-theory / parser) |
| Code generation | emit target code / object file | weighted, categorized, fully explainable AI-likelihood report + JSON export |

---

## 2. Architecture

```
aicb/
├── lexer.py           # Phase 1: tokenizer + rule-based POS tagger
├── grammar.py          # CFG definition (production rules, FIRST sets)
├── parser.py            # Phase 2: recursive-descent parser + panic-mode recovery
├── attributes.py         # Phase 3: attribute grammar evaluator (synth + inherited + voice + clause depth)
├── symbol_table.py        # Phase 4: document-level frequency tables
├── linguistics.py           # Phase 4.5: extended lexical/statistical analysis (NEW)
├── semantic.py                # Phase 5: diagnostics pass
├── scorer.py                    # Phase 6: 12-feature weighted, categorized report synthesis
├── pipeline.py                    # orchestrates all phases end-to-end + batch mode + per-sentence suspicion scoring
├── app.py                           # Flask web app + JSON API (single, batch endpoints)
├── templates/index.html              # single-page UI with 4 modes
├── static/style.css                   # dark, code-editor-inspired theme + print styles
├── static/script.js                    # frontend logic (fetch + rendering + compare/batch/history)
└── requirements.txt
```

### The grammar (`grammar.py` / `parser.py`)

```
S    -> NP VP (CCONJ S | SCONJ S)?
S    -> ADV_CONN COMMA? S        (fronted discourse adverb, e.g. "Moreover, ...")
NP   -> DET? AP* (NOUN|PROPN|PRON|NUM)+ PP*
AP   -> ADJ+
PP   -> PREP NP
VP   -> (MODAL|AUX)* ADV* VERB NP? PP? (ADV_CONN|ADV)*
```

Terminals are POS tags produced by the lexer. The parser is predictive (uses
FIRST sets) and implements **panic-mode error recovery**: unparsable tokens
attach as `LEAF` nodes so one malformed span never aborts analysis of the
rest of the sentence.

### Attribute grammar (`attributes.py`)

Two tree walks (the standard attribute-grammar evaluation algorithm), plus
two extension passes layered on the same tree without touching the grammar:

- **Synthesized** (post-order): `depth`, `width`, `leaf_count`, `token_count`,
  `rule_sig`, `subtree_sig`.
- **Inherited** (pre-order): `depth_from_root`, `sentence_index`, `sibling_position`.
- **Voice** *(new)*: every `VP` node is tagged `passive`/`active` via the
  `AUX(be-form) + VERB(past-participle)` pattern.
- **Clause-nesting depth** *(new)*: synthesized recursively over nested `S`
  productions — how many subordinate/coordinate clauses are chained.

### Extended lexical & statistical pass (`linguistics.py`) — NEW

A second, tree-independent attribute pass over the flat token/text stream:

- **Readability**: Flesch Reading Ease, Flesch-Kincaid Grade (heuristic syllable counter)
- **POS-bigram entropy**: Shannon entropy of the tag-transition stream — a genuine
  information-theoretic predictability measure
- **DEFLATE compression ratio**: `zlib`-compresses the raw text; more redundant/
  predictable text compresses further — a classic entropy proxy
- **Consecutive-sentence similarity**: Jaccard overlap of content words between
  neighboring sentences — catches templated/formulaic phrasing
- **Contraction, stopword, hedge-word, and pronoun-person ratios**

### Features → score (`scorer.py`) — now 12 features across 3 categories

| Category (weight) | Features |
|---|---|
| **Syntactic Structure** (50%) | sentence-length burstiness (14%), parse-tree depth uniformity (7%), grammar-rule repetition (13%), clause-nesting uniformity (6%), POS-bigram predictability (10%) |
| **Lexical & Discourse** (36%) | discourse-connective density (12%), repeated openers (6%), lexical repetition (5%), consecutive-sentence similarity (8%), contraction scarcity (5%) |
| **Information-Theoretic** (14%) | punctuation diversity (4%), compression-based redundancy (10%) |

Every contribution shown in the UI is `feature_score × weight`, fully
auditable. Weak/contested signals (readability, passive-voice ratio, pronoun
person) are deliberately kept **informational only** and excluded from the
score, to keep the score defensible — this is documented directly in the code.

**This is a heuristic, rule-based signal, not a certified detector** — no
detector (including ML-based ones) is 100% reliable.

---

## 3. Design: "Annotated Manuscript"

The whole UI is built around one idea: **a suspected AI text gets treated
like a manuscript under a linguist's red pen** — nothing summarized away,
everything shown at the token level. Concretely:

- **Every token, analyzed** — the centerpiece. Every sentence is rendered as
  an interlinear gloss (the actual linguistic convention: word on top, POS
  tag directly beneath it), with color-coded underlines for discourse
  connectives / hedge words / contractions / hapax words. Click any word to
  open its full analysis record (stopword? frequency in the document?
  syllable count? every flag that was computed for it). A margin column next
  to every sentence lists, in plain language, exactly what was detected
  about that sentence (e.g. "opens with a formal transition phrase",
  "shares notable word overlap with the previous sentence").
- **The score, itemized** — a literal receipt: every one of the twelve
  scored features, grouped by category, with its exact point contribution
  and a plain-English note, adding up to the total at the bottom. Nothing is
  hidden behind a single opaque percentage.
- **Compiler ledger** — the six pipeline phases as a running log, each with
  concrete numbers ("118 tokens produced, 12 distinct POS categories used";
  "7 parse trees built, 2 panic-mode recoveries"), not just phase names.
- **Humanized rewrite** — a track-changes view (strikethrough = removed,
  underline = added) of the rule-based rewrite, plus a before/after score
  comparison and a full transformation log naming the exact rule behind
  every single edit.
- **Deep dive** (collapsed by default) — the symbol table, informational
  linguistic metrics, and full attribute-annotated parse trees remain fully
  available, just not front-and-center.

Four modes are available from the top nav: **Analyze text** (above),
**Compare two texts**, **Batch / paragraphs** (splits on blank lines, scores
each paragraph independently — useful for spotting one AI-written paragraph
inside an otherwise human document), and **History** (saved to
`localStorage`, client-side only). File upload (`.txt`/`.md`) is available
from the Analyze text panel.

### The Humanizer (`humanizer.py`) — rule-based, not AI

A "Humanize This Text" button runs a **deterministic, rule-based rewrite
pipeline** — explicitly NOT a language model, generative AI, or paraphrasing
API — modeled on how a compiler back end applies a fixed sequence of named
optimization passes to code:

1. **connective_softening** — swaps formal discourse connectives (moreover,
   furthermore, additionally, ...) for a rotating pool of casual alternatives
   (also, plus, and, what's more, ...), or drops formal lead-ins like "In
   conclusion,". The rotation is deliberate: mapping every connective to the
   same replacement word would just trade one repetition problem for another,
   so a shared counter cycles through the pool.
2. **contraction_insertion** — "it is" → "it's", "does not" → "doesn't", etc.
3. **sentence_splitting** — breaks long CCONJ-joined compound sentences into
   two, restoring the sentence-length variance ("burstiness") that AI text
   tends to lack.
4. **opener_diversification** — if a sentence repeats the document's most
   over-used opener word and has a fronting prepositional phrase, moves that
   phrase to the front instead.

Every single edit is logged as `{pass, sentence_index, before, after, rule}`
— the "Transformation Log" panel in the UI shows every change with the exact
rule that produced it, and a before/after score comparison (computed by
re-running the *same* transparent scorer on the rewritten text, not asserted)
proves the rewrite actually reduced the AI-likelihood score. No pass invents
content, changes facts, or uses statistical generation — everything is a
closed-class substitution or a grammar-tree-driven restructuring of words the
sentence already contains.


---

## 4. Setup & run

```bash
cd aicb
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in a browser.

## 5. API

- `POST /api/analyze` — `{"text": "..."}` → full JSON report (`pipeline_stages`,
  `symbol_table`, `linguistics`, `diagnostics`, `sentences`, `report`, `passive_ratio`).
  Every token in every sentence carries full transparency flags: `is_stopword`,
  `is_transition_word`, `is_hedge_word`, `is_contraction`, `doc_frequency`,
  `is_hapax`, `syllables`.
- `POST /api/analyze_batch` — `{"text": "..."}` → per-paragraph scores + overall average
- `POST /api/humanize` — `{"text": "..."}` → `{original_score, humanized_text,
  humanized_score, score_delta, edits, passes_summary, original_breakdown,
  humanized_breakdown}`. Rule-based only; no external AI/LLM call is made.
- `GET /api/sample/<ai|human>` — sample text for the demo buttons

## 6. Extending it

- Add new grammar rules in `grammar.py` / `parser.py` (e.g. relative clauses, questions).
- Add new tree attributes in `attributes.py` (the voice/clause-depth passes
  are templates for how to layer a new attribute without touching the grammar).
- Add new statistical signals in `linguistics.py` (informational or scored).
- Add new features/weights in `scorer.py` — keep `WEIGHTS` summing to 1.0
  (there's an `assert` that enforces this).
- Swap the rule-based POS tagger in `lexer.py` for a statistical one if you
  want higher tagging accuracy — the rest of the pipeline is agnostic to
  where POS tags come from.


## Humanize button (local RAG + Ollama, no API key)
1. Install Ollama, then: `ollama pull llama3.2` (or any model, set `AICB_MODEL`) and keep `ollama serve` running.
2. `python app.py`. Clicking "Rewrite in a human voice" runs the rule passes, retrieves the closest human-written passages from `data/human_corpus.txt` (TF-IDF, `rag.py`), has Ollama rewrite in that style, and keeps only rewrites the scorer rates lower (up to 4 rounds, stops at <=2%).
3. Improve results by adding more human text to `data/human_corpus.txt` (blank-line separated). If Ollama is not running it falls back to the rule-only result (~14%).

## Quick start
```
unzip AI_Content_Breaker.zip && cd aicb
./run.sh            # Linux/macOS (Windows: see below)
```
Windows: `py -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt && ollama pull llama3.2 && python app.py`
Open http://localhost:5000

Speed: default model is llama3.2, warmed at startup, max 20 s of LLM time per click. No GPU? use AICB_MODEL=llama3.2:1b (faster) ; more quality: AICB_MODEL=llama3.1:8b AICB_BUDGET=60.

## Calibration (why humanized text can score ~0-5%)
`calibrate.py` runs the scorer on `data/human_corpus.txt` (525 human passages: movie reviews, forum/overheard/wine text from NLTK + hand-written) and stores each feature's human median in `data/calibration.json`.
Each feature is then scored relative to genuine human writing (human median -> 0%). No neural network: it is empirical fitting of thresholds.
More data: download a Kaggle dataset (e.g. IMDB reviews, blog authorship) and run `python add_corpus.py file.csv text_column`.
Verdict thresholds after calibration: >=24 Likely AI, >=16 Mixed, else Likely Human.
