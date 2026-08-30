"""Tutor turns for the L2 multi-turn records.

Each correction in a learner's essay becomes one record whose tutor turn quotes
the wrong text back with the surrounding words and asks for a fix. Two styles are
generated. The point style names the error category, and the
rule style states the convention that governs it and asks the learner to apply
it. Both ask for the corrected fragment alone.
"""

from __future__ import annotations

from studentsim.data.l2.spans import SYMBOL_LABELS, Span

#: One convention per error category, used by the rule style.
RULE_TEMPLATES = {
    "SP": (
        "When you are unsure of how to spell a word, recall the standard English "
        "dictionary spelling. Apply this to the highlighted text."
    ),
    "WC": (
        "Word choice should fit the surrounding context — both grammatically and "
        "semantically. Reconsider whether the highlighted word fits this "
        "sentence's meaning."
    ),
    "AR": (
        "Article rule: use 'a/an' to introduce a countable singular noun for the "
        "first time; use 'the' for previously-introduced or unique referents; use "
        "no article for plural or uncountable nouns when speaking generally. "
        "Apply this to the highlighted phrase."
    ),
    "PR": (
        "Preposition rule: prepositions follow conventions tied to the verb or "
        "noun they pair with — 'in' for being inside a container, 'at' for points "
        "in space/time, 'on' for surfaces or specific days, 'from/to' for "
        "direction. Reconsider the highlighted preposition."
    ),
    "VT": (
        "Verb tense rule: use present-simple for habits or general truths; "
        "past-simple for completed past actions; present-continuous for actions "
        "happening now; present-perfect for past actions with present relevance. "
        "Apply to the highlighted verb."
    ),
    "PL": (
        "Plural rule: most countable nouns add '-s' for plural ('book' → "
        "'books'); some are irregular ('child' → 'children'); mass nouns (water, "
        "advice) do not pluralize. Apply to the highlighted noun."
    ),
    "AG": (
        "Subject-verb agreement: a singular subject takes a singular verb ('she "
        "sings'); a plural subject takes a plural verb ('they sing'). Reconsider "
        "the agreement of the highlighted phrase with its subject."
    ),
    "IS": (
        "Suffix / inflection rule: derivational suffixes change a word's class — "
        "'-ly' makes adverbs, '-tion' makes nouns, '-ize' makes verbs, '-able' "
        "makes adjectives. Apply the right form to the highlighted word."
    ),
    "D": (
        "Redundancy rule: if a word is repeated, contradicts surrounding "
        "structure, or is grammatically unnecessary, it should be removed. Apply "
        "this judgement to the highlighted word."
    ),
    "MW": (
        "Function-word rule: English clauses generally require articles, "
        "subjects, auxiliaries, prepositions, or conjunctions to be complete. If "
        "a position is missing one, supply the appropriate function word."
    ),
    "WO": (
        "Word-order rule: English typically follows Subject-Verb-Object order; "
        "adjectives precede nouns; adverbs of frequency precede the main verb. "
        "Reorder the highlighted segment accordingly."
    ),
    "XC": (
        "Collocation rule: certain words conventionally pair together — e.g., "
        "'make a decision' (not 'do a decision'), 'take a break' (not 'have a "
        "break' in many contexts). Reconsider the highlighted phrase against "
        "standard English collocations."
    ),
    "EX": (
        "Idiomatic-expression rule: fixed expressions follow conventional wording "
        "(e.g., 'on the other hand', 'by the way'). Replace the highlighted "
        "phrase with the standard form."
    ),
    "CO": (
        "Connector rule: discourse connectors (however, therefore, in contrast, "
        "moreover, etc.) signal logical relationships between clauses. Choose the "
        "connector whose meaning matches the relation."
    ),
    "SI": (
        "Syntax rule: a clause must follow standard structure — main verb "
        "agreeing with subject, subordinate clauses with appropriate "
        "subordinators, no fragmented or run-on constructions. Reconsider the "
        "highlighted segment."
    ),
}

_FALLBACK_RULE = "Apply the relevant English convention to the highlighted phrase."

_CLOSING = (
    "Reply with only the corrected text — no explanation, no rewriting the "
    "rest of the essay."
)

#: What a delete correction asks the learner to write, since it corrects to
#: nothing at all.
DELETE_TOKEN = "delete"


def _excerpt(span: Span) -> tuple[str, str]:
    """The quoted excerpt and how the tutor refers to the thing to fix."""
    selection = span.selection.strip()
    before = span.before.strip()
    after = span.after.strip()
    if selection:
        return f"...{before} [WRONG: {selection!r}] {after}...", f"the highlighted phrase {selection!r}"
    return f"...{before} [MISSING WORD HERE] {after}...", "the gap shown above"


def build_point_tutor_turn(span: Span) -> str:
    """Tutor turn that names the error category and points at it."""
    excerpt, target = _excerpt(span)
    label = SYMBOL_LABELS.get(span.symbol, "error")
    return (
        f"Looking at your essay, there is a {label} issue.\n\n"
        f"Excerpt:\n  {excerpt}\n\n"
        f"Please fix {target}. {_CLOSING}"
    )


def build_rule_tutor_turn(span: Span) -> str:
    """Tutor turn that states the convention and asks for it to be applied."""
    excerpt, target = _excerpt(span)
    rule = RULE_TEMPLATES.get(span.symbol, _FALLBACK_RULE)
    return (
        f"Tutor note (rule):\n  {rule}\n\n"
        f"Excerpt:\n  {excerpt}\n\n"
        f"Apply the rule to {target}. {_CLOSING}"
    )


def target_correction(span: Span) -> str:
    """What the learner should write back."""
    return span.correct or DELETE_TOKEN
