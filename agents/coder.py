from typing import Callable
from agents.agent import Agent, AgentConfig, AgentOutput, Context, LLMResponse
from pydantic import BaseModel

from agents.planner import PlannerOutput
from agents.researcher import ResearchOutput
from agents.tools import (
    Error,
    File,
    LintCodeToolResult,
    TestCodeTool,
    TestCodeToolResult,
    Tool,
    EditFileToolResult,
)


class CoderContext(Context):
    planner_output: PlannerOutput | None = None
    researcher_output: ResearchOutput | None = None
    code_changes: list[File] = []
    tests_passed: bool = False
    lint_check_passed: bool = False


class CoderOutput(AgentOutput):
    code_changes: list[File]
    tests_passed: bool
    lint_check_passed: bool


class CoderAgent(Agent):
    name: str = "coder_agent"
    context: CoderContext = CoderContext()

    def is_done(self):
        if self.context.num_iterations >= self.config.max_iterations:
            return True
        if self.context.tests_passed and self.context.lint_check_passed:
            return True
        return False

    def get_output(self) -> CoderOutput:
        return CoderOutput(
            code_changes=self.context.code_changes,
            tests_passed=self.context.tests_passed,
            lint_check_passed=self.context.lint_check_passed,
        )

    def step(self) -> tuple[LLMResponse | None, Error | None]:
        prompt = self.build_prompt(self.config.task_description)
        response, error = self._invoke_llm(prompt)
        self._update_common_context(response, error)

        if error is not None:
            return response, error

        for tool_call in response.tool_calls:
            # make sure that agent does not call itself recursively
            if tool_call.name == "coder_agent":
                error = Error(
                    description="Coder agent should not call itself recursively.",
                )
                return None, error

            if tool_call.name == "edit_file":
                edit_file_result = EditFileToolResult(**tool_call.response)
                self.context.code_changes.append(edit_file_result.file)

            # check if a test_code tool call was made
            # and update the context if found
            if tool_call.name == "test_code":
                test_code_result = TestCodeToolResult(**tool_call.response)
                self.context.tests_passed = test_code_result.error is None

            # check if a lint_code tool call was made
            # and update the context if found
            if tool_call.name == "lint_code":
                lint_code_result = LintCodeToolResult(**tool_call.response)
                self.context.lint_check_passed = lint_code_result.error is None

        return response, None

    def validate_llm_response(self, llm_response: LLMResponse) -> Error | None:
        error = super().validate_llm_response(llm_response)
        # try to nudge the agent to use a tool if it decides to not use any tools
        if not llm_response.tool_calls:
            tools_to_use = []
            if not self.context.code_changes:
                tools_to_use.append("edit_file")

            error = Error(
                description=f"You decided not to use any tools. Please use the available tools to implement the code changes.",
                data={"available_tools": tools_to_use},
            )
        return error

    def build_prompt(self, task_description: str) -> str:
        return f"""
# Task
You are a software engineer that specializes in writing code to solve a problem or task.

Here is the task description:
{task_description}

# Guidelines
- IT IS EXTREMELY IMPORTANT that you write code that is easy to read, understand, and maintain.
- IT IS EXTREMELY IMPORTANT that you do not write code that adds unnecessary changes or complexity.
- IT IS EXTREMELY IMPORTANT that you use the plan to track your progress. Make sure to update the plan as you complete each subtask.
- IT IS EXTREMELY IMPORTANT that you return your response as JSON with 'reasoning', 'tool_calls', and 'confidence' fields.
    - Here is an example of the expected format:
    ```json
    {{
        "reasoning": "I need to write code to solve a bug in the codebase.",
        "tool_calls": [
            {{
                "name": "edit_file",
                "arguments": {{
                    "file_path": "main.py",
                    "content": "<python code here...>"
                }}
            }},
            {{
                "name": "lint_code",
                "arguments": {{
                    "root_dir": "<repository_root_directory>"
                }}
            }},
            {{
                "name": "test_code",
                "arguments": {{
                    "root_dir": "<repository_root_directory>"
                }}
            }}
        ],
        "confidence": 1.0
    }}
    ```

# Context
{self.context.model_dump_json(indent=2)}
"""


class CoderAgentTool(Tool):
    name: str = "coder_agent"

    def run(
        self,
        agent_config: AgentConfig,
        planner_output: PlannerOutput,
        researcher_output: ResearchOutput,
    ) -> CoderOutput:
        context = CoderContext(
            planner_output=planner_output, researcher_output=researcher_output
        )
        agent = CoderAgent(config=agent_config, context=context)
        agent.start()
        return agent.get_output()

    def get_schema(self) -> str | Callable:
        def coder_agent(task_description: str) -> CoderOutput:
            """
            Use this tool to use an agent to write code to solve the task.

            Args:
                task_description: A description that summarizes the task to solve.
            Returns:
                CoderOutput: The output of the coder agent.
            """
            pass

        return coder_agent
