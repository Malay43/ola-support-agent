"""CrewAI crew: Retrieval Agent + Lookup Agent + Response Composer (Tasks 7, 9, 15)."""
import json
import re

import config
from crewai import Agent, Crew, Process, Task

import governance
from mock_llm import MockCrewLLM
from schemas import SupportAnswer
from tools import RagLookupTool, TicketStatusTool


def make_llm():
    if config.MOCK_LLM:
        return MockCrewLLM()
    from crewai import LLM  # optional real model: set MOCK_LLM=false and LLM_MODEL=<provider/model>
    import os

    return LLM(model=os.environ["LLM_MODEL"])


def make_agent(role: str, goal: str, backstory: str, tools: list, llm) -> Agent:
    """Single entry point for building agents - enforces the least-autonomy tool policy."""
    for t in tools:
        governance.assert_tool_allowed(role, t.name)
    return Agent(role=role, goal=goal, backstory=backstory, tools=tools, llm=llm,
                 allow_delegation=False, verbose=False, max_iter=4)


def build_agents(llm=None) -> dict:
    llm = llm or make_llm()
    return {
        "retrieval": make_agent(
            "Retrieval Agent", "Find the support policy that answers the customer's question.",
            "You search the support knowledge base and only report what it contains.", [RagLookupTool()], llm),
        "lookup": make_agent(
            "Lookup Agent", "Look up the status of a specific support ticket.",
            "You are the only agent allowed to read ticket records.", [TicketStatusTool()], llm),
        "composer": make_agent(
            "Response Composer", "Combine policy and ticket facts into one grounded, structured reply.",
            "You write the final answer using only the facts handed to you by your colleagues.", [], llm),
    }


_AGENTS = None


def get_agents():
    global _AGENTS
    if _AGENTS is None:
        _AGENTS = build_agents()
    return _AGENTS


def run_crew(question: str) -> dict:
    """Run the 3-agent crew via .kickoff() and return the validated SupportAnswer plus raw task outputs."""
    a = get_agents()
    t_ret = Task(
        description=f"QUESTION: {question}\nSearch the policy knowledge base for this question using your tool.",
        expected_output="The policy answer with its source ids, or a statement that nothing relevant exists.",
        agent=a["retrieval"])
    t_look = Task(
        description=f"QUESTION: {question}\nIf the question names a ticket id, look that ticket up with your tool.",
        expected_output="The ticket record as JSON, or NO_TICKET_ID if no ticket id was given.",
        agent=a["lookup"])
    t_comp = Task(
        description=f"QUESTION: {question}\nCompose ONE final answer from the retrieval and lookup results only.",
        expected_output="A JSON object matching the SupportAnswer schema.",
        agent=a["composer"], context=[t_ret, t_look], output_pydantic=SupportAnswer)
    crew = Crew(agents=list(a.values()), tasks=[t_ret, t_look, t_comp], process=Process.sequential, verbose=False)
    result = crew.kickoff()

    parsed = getattr(result, "pydantic", None)
    if parsed is None:  # validate in code regardless (Task 9)
        raw = result.raw
        m = re.search(r"\{.*\}", raw, re.S)
        parsed = SupportAnswer.model_validate(json.loads(m.group(0)))
    else:
        parsed = SupportAnswer.model_validate(parsed.model_dump())
    return {
        "answer": parsed,
        "task_outputs": [{"agent": t.agent.role, "output": (t.output.raw if t.output else "")} for t in (t_ret, t_look, t_comp)],
    }
