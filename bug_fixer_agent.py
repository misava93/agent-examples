from agents.agent import AgentConfig
from agents.bug_fixer import BugFixerAgent
from agents.coder import CoderAgentTool
from agents.planner import PlannerAgentTool
from agents.researcher import ResearchAgentTool
from agents.tools import (
    FormatCodeTool,
    LintCodeTool,
    TestCodeTool,
)


def run_bug_fix_agent():
    """
    Starts a bug fix agent that iteratively attempts to fix the bug documented in a ticket.
    """
    print("Please provide ticket ID with bug report (any string is fine): ")
    ticket_id = input()
    ticket_id = ticket_id.strip()
    agent_config = AgentConfig(
        name="bug-fixer",
        model="qwen3:8b",  # gpt-oss:20b
        tools={
            "researcher_agent": ResearchAgentTool(),
            "planner_agent": PlannerAgentTool(),
            "coder_agent": CoderAgentTool(),
            "test_code": TestCodeTool(),
            "lint_code": LintCodeTool(),
            "format_code": FormatCodeTool(),
        },
        max_iterations=50,
        task_description=f"A customer reported a bug via a ticket. ticket_id='{ticket_id}'",
    )
    agent = BugFixerAgent(config=agent_config)
    output = agent.start()
    print("----------------------------------------------")
    print(f"Agent finished. Final Output:\n{output.model_dump_json()}")
    print("----------------------------------------------")


if __name__ == "__main__":
    run_bug_fix_agent()
