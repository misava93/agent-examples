from agents.agent import Agent, AgentConfig, AgentOutput, Context, LLMResponse
from pydantic import BaseModel

from agents.coder import CoderOutput
from agents.planner import PlannerOutput
from agents.researcher import ResearchOutput
from agents.tools import (
    Error,
    File,
    LintCodeToolResult,
    TestCodeTool,
    TestCodeToolResult,
)


class BugFixerContext(Context):
    planner_output: PlannerOutput | None = None
    researcher_output: ResearchOutput | None = None
    coder_output: CoderOutput | None = None
    tests_passed: bool = False
    lint_check_passed: bool = False


class BugFixerOutput(AgentOutput):
    coder_output: CoderOutput


class BugFixerAgent(Agent):
    name: str = "bug_fixer_agent"
    context: BugFixerContext = BugFixerContext()

    def is_done(self):
        if self.context.num_iterations >= self.config.max_iterations:
            return True
        if (
            not self.context.coder_output
            or not self.context.planner_output
            or not self.context.researcher_output
        ):
            return False
        if self.context.tests_passed and self.context.lint_check_passed:
            return True
        return False

    def get_output(self) -> BugFixerOutput:
        return BugFixerOutput(
            coder_output=self.context.coder_output,
        )

    def step(self) -> tuple[LLMResponse | None, Error | None]:
        prompt = self.build_prompt(self.config.task_description)
        response, error = self._invoke_llm(prompt)
        self._update_common_context(response, error)

        if error is not None:
            return response, error

        for tool_call in response.tool_calls:
            if tool_call.name == "bug_fixer_agent":
                # make sure that agent does not call itself recursively
                error = Error(
                    description="Bug fixer agent should not call itself recursively.",
                )
                return None, error
            if tool_call.name == "researcher_agent":
                researcher_output = ResearchOutput(
                    **self.context.sub_agent_responses["researcher_agent"]
                )
                self.context.researcher_output = researcher_output
            if tool_call.name == "planner_agent":
                planner_output = PlannerOutput(
                    **self.context.sub_agent_responses["planner_agent"]
                )
                self.context.planner_output = planner_output
            if tool_call.name == "coder_agent":
                coder_output = CoderOutput(
                    **self.context.sub_agent_responses["coder_agent"]
                )
                self.context.coder_output = coder_output
                self.context.tests_passed = coder_output.tests_passed
                self.context.lint_check_passed = coder_output.lint_check_passed
            if tool_call.name == "test_code":
                if self.context.researcher_output is None:
                    error = Error(
                        description="Researcher agent must be called before calling test_code tool.",
                    )
                    return None, error
                if self.context.planner_output is None:
                    error = Error(
                        description="Planner agent must be called before calling test_code tool.",
                    )
                    return None, error
                if self.context.coder_output is None:
                    error = Error(
                        description="Coder agent must be called before calling test_code tool.",
                    )
                    return None, error
                test_code_result = TestCodeToolResult(**tool_call.response)
                self.context.tests_passed = test_code_result.error is None
            if tool_call.name == "lint_code":
                if self.context.researcher_output is None:
                    error = Error(
                        description="Researcher agent must be called before calling lint_code tool.",
                    )
                    return None, error
                if self.context.planner_output is None:
                    error = Error(
                        description="Planner agent must be called before calling lint_code tool.",
                    )
                    return None, error
                if self.context.coder_output is None:
                    error = Error(
                        description="Coder agent must be called before calling lint_code tool.",
                    )
                    return None, error
                lint_code_result = LintCodeToolResult(**tool_call.response)
                self.context.lint_check_passed = lint_code_result.error is None

        return response, None

    def validate_llm_response(self, llm_response: LLMResponse) -> Error | None:
        return super().validate_llm_response(llm_response)

    def build_prompt(self, task_description: str) -> str:
        return f"""
# Task
You are a software engineer that specializes in fixing bugs in a codebase.

Follow these steps to fix the bug:
1. Research the bug
2. Plan the fix
3. Implement the fix
4. Validate that the fix works:
    - Run tests and make sure they pass
    - Run lint checks and make sure there are no errors

Here is the task description:
{task_description}

# Guidelines
- IT IS EXTREMELY IMPORTANT that you use the 'researcher_agent' tool to research the bug and gather more information.
- IT IS EXTREMELY IMPORTANT that you use the 'planner_agent' tool to plan the fix.
- IT IS EXTREMELY IMPORTANT that you use the 'coder_agent' tool to implement the fix.
- IT IS EXTREMELY IMPORTANT that you use the 'test_code' tool to validate that the fix works.
- IT IS EXTREMELY IMPORTANT that you use the 'lint_code' tool to validate that the fix does not introduce new linting errors.
- IT IS EXTREMELY IMPORTANT that you use the 'format_code' tool to format the code to follow linting standards.
- IT IS EXTREMELY IMPORTANT that you write tests to reproduce the bug before you implement the fix. Make sure to incorporate this in your plan.
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
