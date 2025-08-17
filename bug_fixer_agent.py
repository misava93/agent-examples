import difflib
import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

import ollama
from pydantic import BaseModel

from agents.agent import AgentConfig
from agents.coder import CoderAgentTool
from agents.planner import PlannerAgentTool
from agents.researcher import ResearchAgentTool
from agents.tools import EditFileTool, FindFilesTool, FormatCodeTool, GetTicketTool, LintCodeTool, ReadFileTool, SearchWebTool, TestCodeTool
from agents.bug_fixer import BugFixerAgent

def run_bug_fix_agent():
    """
    Starts a bug fix agent that iteratively attempts to fix the bug documented in a ticket.
    """
    print("Please provide ticket ID with bug report (any string is fine): ")
    ticket_id = input()
    ticket_id = ticket_id.strip()
    agent_config = AgentConfig(
        name="bug-fixer",
        model="gpt-oss:20b",
        tools={
            "edit_file": EditFileTool(),
            "researcher_agent": ResearchAgentTool(),
            "planner_agent": PlannerAgentTool(),
            "coder_agent": CoderAgentTool(),
            "test_code": TestCodeTool(),
            "lint_code": LintCodeTool(),
            "format_code": FormatCodeTool(),
        },
        max_iterations=50,
        task_description=f"Fix the bug documented in ticket. ticket_id='{ticket_id}'",
    )
    agent = BugFixerAgent(config=agent_config)
    output = agent.start()
    print("----------------------------------------------")
    print(f"Agent finished. Final Output:\n{output.model_dump_json()}")
    print("----------------------------------------------")

if __name__ == "__main__":
    run_bug_fix_agent()
