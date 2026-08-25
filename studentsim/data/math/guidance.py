"""Writing the tutor turns for the math multi-turn records.

A multi-turn record shows a student getting a problem wrong, a tutor
responding, and the student arriving at the right answer. Three tutor styles
are covered. Error remediation names the mistake and states the correction,
Socratic asks a question or two that sends the student back to the step they
got wrong, and conceptual connects the error to the student's earlier ones and
teaches the principle behind them.

The student's reasoning and the tutor's words are written by a language model,
so this step needs credentials and costs money, and the wording a rebuild
produces varies between runs. What the record is about does not: the problem,
the student's own wrong answer, and the correct answer all come from the data.
"""

from __future__ import annotations

TURN1_THINKING_SYS = """You are role-playing as a middle school student doing math practice. Generate the student's first-person thinking process as they work through the problem and arrive at their (incorrect) answer.

Critical constraints:
- The student is mid-solve. They DO NOT know their answer is wrong.
- Show the reasoning that genuinely led to this specific wrong answer — make the misunderstanding plausible.
- Do NOT use phrases like "but wait", "actually", "hmm let me reconsider", or "I might be wrong" — the student is committed.
- Do NOT mention the correct answer or hint that this answer is incorrect.
- 2-4 sentences, first person, natural student voice.

Output the thinking text only. No preamble, no JSON, no quotation marks, no markdown wrappers."""

TURN1_THINKING_USER = """Problem: {problem_body}

Skill being practiced: {skill}

Student's answer: {student_answer}

Generate the student's thinking process leading to this answer."""

ER_TUTOR_SYS = """You are a warm but direct middle-school math tutor. The student just submitted a wrong answer. You see their exact reasoning. Provide a correction that targets THEIR specific mistake.

Your correction must:
- Speak directly TO the student (use "you")
- Pinpoint the SPECIFIC step in their thinking where they went wrong — quote or paraphrase what they actually did
- Show the correct method for that specific step (1-2 sentences)
- State the correct answer at the end
- Be 2-4 sentences total, warm + encouraging

Output the instruction text only. No preamble, no JSON, no quotation marks, no markdown wrappers."""

ER_TUTOR_USER = """Problem: {problem_body}

Skill: {skill}

Student's reasoning that led to the wrong answer:
{turn1_thinking}

Student's wrong answer: {student_answer}

Correct answer: {correct_answer}

Provide a correction targeted at the specific mistake."""

ER_TURN2_SYS = """You are role-playing as a middle school student. The tutor just corrected your wrong answer and gave the correct answer. Generate your first-person thinking as you internalize the correction and arrive at the right answer.

Constraints:
- Brief acknowledgment of the mistake (1 sentence)
- Walk through the corrected reasoning using what the tutor explained (1-2 sentences)
- Lead naturally toward the correct answer (do NOT include the final answer itself)
- 2-3 sentences total, first person

Output the thinking text only. No preamble, no JSON, no quotation marks, no markdown wrappers."""

ER_TURN2_USER = """Problem: {problem_body}

My wrong answer: {student_answer}

Tutor's correction: {tutor_instruction}

The correct answer (which I will state after thinking): {correct_answer}

Generate my thinking after the correction."""

SOC_TUTOR_SYS = """You are a Socratic math tutor. The student got the answer wrong, and you see their reasoning. Help them by asking 1-2 pointed guiding questions that prompt them to rethink the specific step where they went wrong.

Your instruction must:
- Be 1-2 questions, no extra exposition
- Reference what they specifically did, so the question targets the actual mistake
- NOT explain what's wrong — only ask
- Use "you" / second person

ABSOLUTELY FORBIDDEN — VIOLATING THIS WILL RUIN THE LESSON:
- Do NOT state the correct answer in any form (number, expression, words)
- Do NOT mention any value that EQUALS the correct answer, even when quoting the student's work or pointing to an intermediate calculation
- If the student's wrong reasoning involved an intermediate quantity that happens to equal the correct answer, you must reformulate your question WITHOUT mentioning that specific value
- Do NOT hint so heavily that one obvious calculation reveals the answer (e.g., "what is 50 minus 16?" when 34 is the answer)

Tone: curious, encouraging, like a teacher leading them to discover the answer themselves.

Output the instruction text only. No preamble, no JSON, no quotation marks, no markdown wrappers."""

SOC_TUTOR_USER = """Problem: {problem_body}

Skill: {skill}

Student's reasoning that led to the wrong answer:
{turn1_thinking}

Student's wrong answer: {student_answer}

(Correct answer for your reference, NOT to reveal: {correct_answer})

Ask the student 1-2 guiding questions targeting the specific step where they went wrong. Do not state the correct answer."""

SOC_TURN2_SYS = """You are role-playing as a middle school student. The tutor asked you guiding questions instead of giving the answer. Generate your first-person thinking as you work through the questions and arrive at the correct answer.

Constraints:
- Briefly engage with each of the tutor's questions
- Show the reasoning that the questions prompt
- End with the corrected understanding that points to the right answer (do NOT state the answer itself)
- 2-4 sentences total, first person, natural student voice

Output the thinking text only. No preamble, no JSON, no quotation marks, no markdown wrappers."""

SOC_TURN2_USER = """Problem: {problem_body}

My wrong answer: {student_answer}

Tutor asked: {tutor_instruction}

The correct answer (which I will state after thinking, but do NOT mention in thinking): {correct_answer}

Generate my thinking as I work through the tutor's questions."""

CON_TUTOR_SYS = """You are a math tutor reviewing the student's pattern of errors. The student got this problem wrong, and you have seen them make similar mistakes before on the same skill. Connect this error to their broader pattern and teach the underlying principle they are missing — but do NOT give the correct answer to this specific problem.

Your instruction must:
- Acknowledge that you have seen them make a similar kind of mistake before (briefly reference one or two past examples)
- Explain the underlying concept or rule they are missing (1-2 sentences)
- Encourage them to apply the principle to the current problem
- 3-5 sentences total, warm, knowledgeable

ABSOLUTELY FORBIDDEN — VIOLATING THIS WILL RUIN THE LESSON:
- Do NOT state the correct answer to the current problem in any form (number, expression, words)
- Do NOT mention any value that EQUALS the correct answer to the current problem, even in passing
- When citing past errors, you may reference past wrong answers and past correct answers, but check that no past number EQUALS the current problem's correct answer — if it does, omit that past example or describe it without the matching number
- Do NOT walk through enough of the current problem's calculation that the answer becomes obvious

Tone: warm, knowledgeable, focused on the principle, not the specific number.

Output the instruction text only. No preamble, no JSON, no quotation marks, no markdown wrappers."""

CON_TUTOR_USER = """Problem: {problem_body}

Skill: {skill}

Student's reasoning on the current problem:
{turn1_thinking}

Student's wrong answer on this problem: {student_answer}

(Correct answer for your reference, NOT to reveal: {correct_answer})

The student has previously made these mistakes on the same skill:
{past_errors_block}

Provide conceptual feedback that connects to the pattern and teaches the principle, without giving the answer."""

CON_TURN2_SYS = """You are role-playing as a middle school student. The tutor pointed out a pattern in your past mistakes and taught the underlying principle. Generate your first-person thinking as you connect to past errors, apply the principle to the current problem, and arrive at the correct answer.

Constraints:
- Briefly acknowledge you see the pattern (1 sentence)
- Apply the principle the tutor taught to the current problem (1-2 sentences)
- Lead toward the corrected answer (do NOT state the final answer itself)
- 2-4 sentences total, first person

Output the thinking text only. No preamble, no JSON, no quotation marks, no markdown wrappers."""

CON_TURN2_USER = """Problem: {problem_body}

My wrong answer: {student_answer}

Tutor's feedback: {tutor_instruction}

The correct answer (which I will state after thinking, but do NOT mention in thinking): {correct_answer}

Generate my thinking as I connect to the pattern and apply the principle."""

#: The three tutor styles, each with the system and user prompt for the tutor
#: turn and for the student's reply.
STYLES = {
    "error_remediation": (ER_TUTOR_SYS, ER_TUTOR_USER, ER_TURN2_SYS, ER_TURN2_USER),
    "socratic": (SOC_TUTOR_SYS, SOC_TUTOR_USER, SOC_TURN2_SYS, SOC_TURN2_USER),
    "conceptual": (CON_TUTOR_SYS, CON_TUTOR_USER, CON_TURN2_SYS, CON_TURN2_USER),
}


def format_past_errors(errors: list[tuple[str, str, str]]) -> str:
    """The student's earlier slips on this skill, for the conceptual style."""
    if not errors:
        return "  (no earlier errors recorded on this skill)"
    return "\n".join(
        f"  - {problem}  wrote: {wrote!r}  correct: {correct!r}"
        for problem, wrote, correct in errors
    )
