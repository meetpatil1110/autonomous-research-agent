import json

from agent.planner import generate_plan, replan


def _fake_llm(payload: dict):
    def call(_system: str, _user: str) -> str:
        return json.dumps(payload)

    return call


def test_generate_plan_returns_requested_steps():
    fake = _fake_llm({"steps": ["sub-question a", "sub-question b", "sub-question c"]})
    steps = generate_plan(fake, "What is the impact of X on Y?")
    assert steps == ["sub-question a", "sub-question b", "sub-question c"]


def test_generate_plan_clamps_oversized_plans():
    fake = _fake_llm({"steps": [f"step {i}" for i in range(10)]})
    steps = generate_plan(fake, "question")
    assert len(steps) == 4


def test_generate_plan_falls_back_to_question_when_empty():
    fake = _fake_llm({"steps": []})
    steps = generate_plan(fake, "question")
    assert steps == ["question"]


def test_replan_returns_additional_steps():
    fake = _fake_llm({"additional_steps": ["one more angle"]})
    steps = replan(fake, "question", [{"step_id": 0, "content": "thin result"}])
    assert steps == ["one more angle"]
