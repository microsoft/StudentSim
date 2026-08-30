"""Rendering a learner's prompt.

Two shapes of profile block are rendered. Single-turn records
carry the compact block: how often each error category shows up, and the
learner's most recent essays in time order. Multi-turn records carry the
detailed block: the same counts plus worked wrong-to-right pairs, and past
essays picked for their similarity to the task at hand. Both shapes end with
the lesson task and the key words drawn from it.
"""

from __future__ import annotations

import re

from studentsim.data.l2.profile import LearnerProfile
from studentsim.data.l2.spans import SYMBOL_LABELS

#: Past essays quoted back to the learner.
RECENT_ESSAYS = 3

#: How much of each quoted essay is shown, in characters.
ESSAY_PREVIEW_CHARS = 200

_KEYWORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")

_STOPWORDS = frozenset("""
the a an of and to in is it you that this for he she we they i be was were am are
been being have has had do does did will would shall should can could may might
must on at by from with as but or if then else also so not no yes my your his
her our their its me him us them mine yours hers ours theirs which who whom whose
what when where why how very just only even still much many more most some any
all each every other another such same own both few several any all
about above across after against along among around before behind below beneath
beside between beyond but during except for from in inside into like near of off
on onto out outside over past since through throughout till to toward under
underneath until up upon with within without
go went going gone goes get got getting take took taken make made making see saw
seen come came coming say said saying think thought thinking know knew known want
wanted wanting need needed needing put give gave given let lets letting use used
using find found finding tell told telling become became becoming feel felt
ask asked asking try tried trying call called calling work worked working seem
seemed seeming look looked looking
i'm i've i'll i'd you're you've you'll you'd he's she's it's we're we've
they're they've don't doesn't didn't won't wouldn't can't couldn't shouldn't
isn't aren't wasn't weren't haven't hasn't hadn't
also however therefore moreover indeed furthermore meanwhile nevertheless
yet still though although while because since unless whether
yes no maybe perhaps okay ok hi hello hey thanks thank please sorry well
""".split())


def extract_keywords(essay: str, max_keywords: int | None = None) -> list[str]:
    """Content words from the essay, in the order the learner used them.

    These go into the prompt as the key words the lesson asked for. The count
    scales with essay length so that short and long tasks leak a comparable
    share of the text.
    """
    if not essay:
        return []
    if max_keywords is None:
        max_keywords = max(3, min(7, len(essay.split()) // 10))

    picked: list[str] = []
    seen: set[str] = set()
    for token in _KEYWORD_RE.findall(essay):
        low = token.lower()
        if low in _STOPWORDS or low in seen:
            continue
        seen.add(low)
        picked.append(token)
        if len(picked) >= max_keywords:
            break
    return picked


def _pick_recent(earlier: list[dict], current: dict, *, by_similarity: bool) -> list[dict]:
    """The past essays quoted in the prompt, most recent last.

    Chronological order takes the last few essays the learner wrote. Selection
    by similarity prefers essays on the same lesson topic, then the same level
    and a neighbouring unit, then the same level, and falls back to the recent
    tail when none of those fills the quota.
    """
    if not earlier:
        return []
    if not by_similarity:
        return earlier[-RECENT_ESSAYS:]

    topic = current.get("topic_id")
    if topic:
        same_topic = [e for e in earlier if e.get("topic_id") == topic]
        if len(same_topic) >= RECENT_ESSAYS:
            return same_topic[-RECENT_ESSAYS:]

    level = current.get("level")
    if level:
        unit = current.get("unit") or 0
        adjacent = [
            e
            for e in earlier
            if e.get("level") == level and abs((e.get("unit") or 0) - unit) <= 1
        ]
        if len(adjacent) >= RECENT_ESSAYS:
            return adjacent[-RECENT_ESSAYS:]
        same_level = [e for e in earlier if e.get("level") == level]
        if len(same_level) >= RECENT_ESSAYS:
            return same_level[-RECENT_ESSAYS:]

    return earlier[-RECENT_ESSAYS:]


def _render_recent(essays: list[dict]) -> str:
    lines = []
    for essay in essays:
        body = (essay["original"] or "").replace("\n", " ").strip()
        if len(body) > ESSAY_PREVIEW_CHARS:
            body = body[:ESSAY_PREVIEW_CHARS] + "..."
        topic = essay["topic"] or "<unknown topic>"
        lines.append(f"  [{topic}] (graded {essay['grade']}): {body!r}")
    return "\n".join(lines)


def _render_error_examples(examples: dict[str, list[tuple[str, str]]]) -> str:
    lines = []
    for symbol, pairs in examples.items():
        label = SYMBOL_LABELS.get(symbol, symbol)
        shown = ", ".join(f"{wrong!r} → {right!r}" for wrong, right in pairs)
        lines.append(f"  - {label}: e.g. {shown}")
    return "\n".join(lines)


def build_user_message(
    *,
    learner_name: str,
    profile: LearnerProfile | None,
    current: dict,
    earlier: list[dict],
    detailed_errors: bool,
) -> str:
    """The prompt a simulator answers as this learner.

    ``detailed_errors`` selects the shape: ``False`` for single-turn records and
    ``True`` for multi-turn ones.
    """
    lines = ["You are simulating an English-learner writing an essay.", ""]
    lines.append("Learner profile:")
    lines.append(f"  Name: {learner_name}")

    if profile is None:
        lines.append("  This is their first writing — no prior history yet.")
    else:
        nationality = profile.nationality or "?"
        cefr = profile.cefr or "?"
        level = profile.current_level or "?"
        lines.append(
            f"  Nationality: {nationality}  Current EFCAMDAT level: {level}  (~CEFR {cefr})"
        )
        summary = (
            f"  Essays written so far: {profile.n_essays}, "
            f"avg ~{profile.avg_wordcount:.0f} words/essay"
        )
        if profile.avg_grade is not None:
            summary += f", avg grade {profile.avg_grade:.0f}"
        lines.append(summary)

        examples = _render_error_examples(profile.error_examples) if detailed_errors else ""
        if examples:
            counts = ", ".join(
                f"{SYMBOL_LABELS.get(s, s)}({n}×)" for s, n in profile.common_errors
            )
            lines.append(f"  Error frequency: {counts}")
            lines.append("  Concrete error examples (this learner wrote → was corrected):")
            lines.append(examples)
        elif profile.common_errors:
            counts = ", ".join(
                f"{SYMBOL_LABELS.get(s, s)}({n})" for s, n in profile.common_errors
            )
            lines.append(f"  Common error patterns: {counts}")

        lines.append("")
        heading = (
            "Recent essays on similar tasks (most recent last):"
            if detailed_errors
            else "Recent essays (most recent last):"
        )
        lines.append(heading)
        recent = _pick_recent(earlier, current, by_similarity=detailed_errors)
        lines.append(_render_recent(recent) if recent else "  (no prior essays — first writing)")

    keywords = extract_keywords(current.get("original") or "")
    lines.append("")
    lines.append("Now write a response to this lesson task:")
    lines.append(f"  Topic: {current.get('topic', '?')}")
    lines.append(f"  Level: {current.get('level', '?')} (unit {current.get('unit', '?')})")
    if keywords:
        lines.append(f"  Use these key words from the lesson: {', '.join(keywords)}")
    lines.append("")
    lines.append(
        "Write the response as this learner would — including the same kinds of "
        "small mistakes they typically make. Do not over-correct."
    )
    return "\n".join(lines)
