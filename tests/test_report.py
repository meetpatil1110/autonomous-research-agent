from agent.report import generate_report


def _fake_llm(response: str):
    def call(_system: str, _user: str) -> str:
        return response

    return call


def test_generate_report_appends_references_section():
    findings = [
        {
            "step_id": 0,
            "content": "Tokyo has 14 million people",
            "sources": ["https://a.example"],
        },
        {
            "step_id": 1,
            "content": "Growth rate is 0.5%",
            "sources": ["https://b.example", "https://a.example"],
        },
    ]
    fake = _fake_llm("Tokyo's population is large [1] and growing slowly [2][1].")
    report = generate_report(fake, "How big is Tokyo?", findings)
    assert "## References" in report
    assert "[1] https://a.example" in report
    assert "[2] https://b.example" in report


def test_generate_report_handles_calculator_source():
    findings = [{"step_id": 0, "content": "2 + 2 = 4", "sources": ["calculator"]}]
    fake = _fake_llm("The answer is 4 [1].")
    report = generate_report(fake, "What is 2+2?", findings)
    assert "[1] calculator" in report


def test_generate_report_with_no_sources_omits_references_section():
    fake = _fake_llm("No information was found.")
    report = generate_report(fake, "Unanswerable question", [])
    assert "## References" not in report


def test_generate_report_normalizes_fullwidth_brackets():
    findings = [{"step_id": 0, "content": "x", "sources": ["https://a.example"]}]
    fake = _fake_llm("The answer is 4【1】.")
    report = generate_report(fake, "question", findings)
    assert "[1]" in report
    assert "【" not in report and "】" not in report
