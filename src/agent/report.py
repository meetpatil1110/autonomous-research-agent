"""Synthesizes a cited research report from accumulated findings.

Citation numbering is assigned in code, not left to the LLM: the model is
given findings pre-tagged with their reference numbers and just needs to
weave them into prose, while the References section is built deterministically
from the same source list. Getting the bookkeeping right doesn't need an LLM.
"""
from __future__ import annotations

from .planner import LLMCall
from .state import Finding

_REPORT_SYSTEM_PROMPT = """You are a research analyst. You are given a research \
question and a list of findings, each already tagged with reference numbers \
like [1] or [1][2] pointing at sources.

Write a structured report answering the question:
- Use markdown with a few section headings appropriate to the question.
- Every factual claim must carry the [n] marker(s) from the finding(s) it came from.
- Use plain ASCII square brackets for markers, e.g. [1], never full-width or
  any other bracket style.
- Do not invent new reference numbers or renumber the ones given.
- Do not write a References/Sources section yourself — that is appended separately.
Respond with the report body only, no preamble."""

# Same rationale as planner._MAX_FINDING_CHARS: Groq's free-tier TPM cap is
# small enough that unbounded finding content blows the budget in one request.
_MAX_FINDING_CHARS = 400


def _build_source_index(findings: list[Finding]) -> tuple[list[str], dict[str, int]]:
    sources: list[str] = []
    index: dict[str, int] = {}
    for finding in findings:
        for source in finding["sources"]:
            if source not in index:
                index[source] = len(sources) + 1
                sources.append(source)
    return sources, index


def _format_findings_with_citations(
    findings: list[Finding], index: dict[str, int]
) -> str:
    lines = []
    for finding in findings:
        markers = "".join(f"[{index[s]}]" for s in finding["sources"]) or "[uncited]"
        content = finding["content"][:_MAX_FINDING_CHARS]
        lines.append(f"- {markers} {content}")
    return "\n".join(lines)


def generate_report(llm_call: LLMCall, question: str, findings: list[Finding]) -> str:
    sources, index = _build_source_index(findings)
    findings_block = _format_findings_with_citations(findings, index)
    user_prompt = f"Research question: {question}\n\nFindings:\n{findings_block}"
    body = llm_call(_REPORT_SYSTEM_PROMPT, user_prompt)
    # Observed gpt-oss-120b occasionally use full-width brackets for markers
    # despite the prompt; normalize rather than rely on instruction-following
    # for something this mechanical.
    body = body.replace("【", "[").replace("】", "]")

    if not sources:
        return body
    references = "\n".join(f"[{i}] {url}" for i, url in enumerate(sources, start=1))
    return f"{body}\n\n## References\n{references}"
