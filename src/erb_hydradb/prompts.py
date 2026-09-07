"""Answer-generation prompts, verbatim from the published run.

Version ``two-pass-v1``. Do not edit in place: a change to any of these strings
changes the generation fingerprint and therefore the run. Add a new version
constant instead.
"""

PROMPTS_VERSION = "two-pass-v1"

PASS1_PROMPT = """You are an expert analyst. Answer the employee's question using ONLY the retrieved company documents provided below.

CRITICAL REQUIREMENTS:
- This answer will be graded fact-by-fact against a gold reference derived from these same documents. Your score depends on EXHAUSTIVELY and PRECISELY stating EVERY specific fact from the documents that is relevant to the question.
- Include every number, unit, threshold, limit, version, ID (ticket IDs, job IDs, task/run IDs), dashboard name, Slack channel, tool/command (exact flags), file name, environment name, duration, cadence, percentage, error code, and person/team role.
- State exact procedural steps and their correct ORDER (do not reorder or swap them). If the docs describe primary vs fallback paths, keep that priority ordering (e.g. try forward recovery FIRST before rollback).
- Do NOT invent, soften, or generalize any fact. If a value is specific in the docs, give the exact value; never replace it with a range or a vague synonym.
- If the docs do not contain a part of the answer, say so explicitly for that part.
- If you are genuinely uncertain whether the docs support a claim, do not state it.
- Be self-contained and professional.

## Retrieved documents
{context}

## Question
{question}

## Answer
"""

CRITIQUE_PROMPT = """You are grading a candidate answer against the authoritative retrieved company documents. Your job is to find concrete gaps so the answer can be corrected.

Compare the candidate answer to the documents. Identify:
1. MISSING: every specific fact in the documents relevant to the question {question} that the answer does NOT state (numbers, units, thresholds, IDs, dates, durations, percentages, commands/flags, names, error codes, exact procedural steps and their order).
2. WRONG/CONTRADICTED: every place the answer states a fact differently from or contradicting the documents (wrong value, wrong ordering, wrong priority between primary and fallback paths, invented detail, softened/generalized specific fact).
3. UNSUPPORTED: any claim in the answer not supported by the documents.

Output a strict JSON object:
{{"missing": ["...", "..."], "wrong": ["...", "..."], "unsupported": ["...", "..."]}}
List every concrete item; be exhaustive and precise. Do not explain; just the arrays.

## Retrieved documents
{context}

## Question
{question}

## Candidate answer
{answer}

## JSON
"""

PASS2_PROMPT = """You are an expert analyst. Rewrite and correct the candidate answer using ONLY the retrieved company documents and the critique below.

Requirements:
- Incorporate EVERY item from the critique's "missing" and "wrong" lists into the corrected answer, stated precisely and correctly (exact numbers, IDs, thresholds, commands, error codes, exact step ordering, primary-vs-fallback priority).
- Remove anything the critique flagged as "unsupported" unless clearly supported by the documents.
- Continue to state EVERY specific fact from the documents relevant to the question, exhaustively and precisely, without inventing details.
- Keep the corrected answer self-contained and professional.

## Retrieved documents
{context}

## Question
{question}

## Candidate answer (to correct)
{answer}

## Critique
{critique}

## Corrected Answer
"""
