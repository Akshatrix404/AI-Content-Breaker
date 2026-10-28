"""
ir_gen.py -- INTERMEDIATE REPRESENTATION & CODE OPTIMIZATION
================================================================
Classic compiler middle-end, repurposed: instead of generating three-
address code from an arithmetic-expression parse tree, we generate a
small triple-based IR from each sentence's syntax tree -- one ASSERT
triple per clause (subject, verb, object) and one MOD triple per
modifier attached to a noun or verb. Two textbook optimization passes
then run over the whole document's IR stream, exactly the way a real
optimizer would run passes over a basic block:

  - Common Subexpression Elimination (CSE): an IR triple that is byte-
    identical to one seen earlier is marked ELIMINATED rather than
    deleted (so the before/after is inspectable) -- real CSE, just over
    propositions instead of arithmetic sub-expressions.
  - Dead-Code Elimination (DCE): a MOD triple whose modifier is a
    generic, information-free filler word (from a fixed closed list) is
    marked DEAD -- the semantic-content analogue of removing a computed
    value nothing downstream uses.

Every pass is logged with a name and a reason, same as humanizer.py's
transformation log, so nothing here is a black box either.
"""

DEAD_MODIFIERS = {
    "very", "really", "quite", "rather", "fairly", "simply", "basically",
    "actually", "literally", "certainly", "definitely", "clearly",
    "various", "numerous", "significant", "important", "notable",
    "considerable", "substantial", "overall", "generally", "largely",
}


def _head_word(np_node):
    """Return the head noun token (last NOUN/PROPN/PRON/NUM child) of an NP node."""
    if np_node is None:
        return None
    head = None
    for c in np_node.children:
        if c.symbol in ("NOUN", "PROPN", "PRON", "NUM") and c.token:
            head = c.token.value
    return head


def _adjectives(np_node):
    if np_node is None:
        return []
    out = []
    for c in np_node.children:
        if c.symbol == "AP":
            for a in c.children:
                if a.token:
                    out.append(a.token.value)
    return out


def _find(node, symbol):
    for c in node.children:
        if c.symbol == symbol:
            return c
    return None


def extract_sentence_ir(tree, sentence_index):
    """Walks one sentence's parse tree and emits IR triples (as dicts)."""
    instructions = []
    np_node = _find(tree, "NP")
    vp_node = _find(tree, "VP")

    subject = _head_word(np_node)
    for adj in _adjectives(np_node):
        instructions.append({"op": "MOD", "arg1": subject, "arg2": adj.lower(),
                              "sentence_index": sentence_index, "note": f"modifier '{adj}' on subject '{subject}'"})

    if vp_node is not None:
        verb_node = _find(vp_node, "VERB")
        verb = verb_node.token.value if (verb_node and verb_node.token) else None
        adverbs = [c.token.value for c in vp_node.children if c.symbol == "ADV" and c.token]
        for adv in adverbs:
            instructions.append({"op": "MOD", "arg1": verb, "arg2": adv.lower(),
                                  "sentence_index": sentence_index, "note": f"modifier '{adv}' on verb '{verb}'"})
        obj_np = _find(vp_node, "NP")
        obj = _head_word(obj_np)
        for adj in _adjectives(obj_np):
            instructions.append({"op": "MOD", "arg1": obj, "arg2": adj.lower(),
                                  "sentence_index": sentence_index, "note": f"modifier '{adj}' on object '{obj}'"})
        if subject or verb or obj:
            instructions.append({"op": "ASSERT", "arg1": (subject or "").lower() or None,
                                  "arg2": (verb or "").lower() or None,
                                  "result": (obj or "").lower() or None,
                                  "sentence_index": sentence_index,
                                  "note": f"{subject or '?'} --{verb or '?'}--> {obj or '(none)'}"})
    return instructions


def _sig(instr):
    """The part of an instruction that defines 'identical computation' for CSE."""
    if instr["op"] == "ASSERT":
        return ("ASSERT", instr["arg1"], instr["arg2"], instr["result"])
    return ("MOD", instr["arg1"], instr["arg2"])


def optimize_ir(instructions):
    """
    Runs CSE then DCE over the full instruction stream, in two visible
    passes (mirrors a real optimizing compiler's pass pipeline). Returns
    the annotated instruction list plus a summary of what each pass did.
    """
    seen = {}
    cse_eliminated = 0
    for instr in instructions:
        sig = _sig(instr)
        if sig in seen and all(v is not None for v in sig[1:]):
            instr["status"] = "ELIMINATED"
            instr["reason"] = f"common subexpression of instruction #{seen[sig] + 1} (identical proposition already computed)"
            cse_eliminated += 1
        else:
            instr["status"] = "LIVE"
            instr["reason"] = ""
            seen[sig] = instructions.index(instr)

    dce_removed = 0
    for instr in instructions:
        if instr["status"] == "LIVE" and instr["op"] == "MOD" and instr["arg2"] in DEAD_MODIFIERS:
            instr["status"] = "DEAD"
            instr["reason"] = f"'{instr['arg2']}' is a generic, information-free filler modifier"
            dce_removed += 1

    total = len(instructions)
    live = sum(1 for i in instructions if i["status"] == "LIVE")
    return {
        "instructions": instructions,
        "total": total,
        "live": live,
        "cse_eliminated": cse_eliminated,
        "dce_removed": dce_removed,
        "redundancy_ratio": round(cse_eliminated / total, 3) if total else 0.0,
        "dead_modifier_ratio": round(dce_removed / total, 3) if total else 0.0,
        "passes_summary": [
            {"pass": "common_subexpression_elimination",
             "detail": f"{cse_eliminated} of {total} IR instruction(s) were exact repeats of an earlier proposition"},
            {"pass": "dead_code_elimination",
             "detail": f"{dce_removed} of {total} MOD instruction(s) carried a generic filler modifier with no discriminative content"},
        ],
    }
