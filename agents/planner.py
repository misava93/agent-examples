from typing import Callable

from agents.agent import Agent, AgentConfig, Context, LLMResponse, PlannerOutput
from agents.researcher import ResearchOutput
from agents.tools import Error, Tool, EditFileToolResult


class PlannerContext(Context):
    researcher_output: ResearchOutput | None = None
    plan_file: str = ""
    plan_file_content: str = ""


class PlannerAgent(Agent):
    name: str = "planner_agent"
    context: PlannerContext = PlannerContext()

    def is_done(self):
        if self.context.num_iterations >= self.config.max_iterations:
            return True
        if self.context.plan_file and self.context.plan_file_content:
            return True
        return False

    def get_output(self) -> PlannerOutput:
        return PlannerOutput(
            plan_file_path=self.context.plan_file,
        )

    def step(self) -> tuple[LLMResponse | None, Error | None]:
        prompt = self.build_prompt(self.config.task_description)
        response, error = self._invoke_llm(prompt)
        self._update_common_context(response, error)

        if error is not None:
            return response, error

        for tool_call in response.tool_calls:
            # make sure that agent does not call itself recursively
            if tool_call.name == "planner_agent":
                error = Error(
                    description="Planner agent should not call itself recursively.",
                )
                return None, error

            # check if an edit_file tool call was made with the file_path "PLAN.md"
            # and update the context if found
            if (
                tool_call.name == "edit_file"
                and tool_call.arguments["file_path"] == "PLAN.md"
            ):
                edit_file_result = EditFileToolResult(**tool_call.response)
                self.context.plan_file = edit_file_result.file.path
                self.context.plan_file_content = edit_file_result.file.content

        return response, error

    def validate_llm_response(self, llm_response: LLMResponse) -> Error | None:
        error = super().validate_llm_response(llm_response)

        # try to nudge the agent to use a tool if it decides to not use any tools
        if not llm_response.tool_calls:
            tools_to_use = []
            if not self.context.plan_file_content:
                tools_to_use.append("edit_file")

            error = Error(
                description=f"You decided not to use any tools. Please use the available tools to create a plan.",
                data={"available_tools": tools_to_use},
            )
        return error

    def build_prompt(self, task_description: str) -> str:
        return f"""
# Task
You are a software engineer that specializes in creating a plan to solve a problem or task.

Here is the task description:
{task_description}

# Guidelines
- IT IS EXTREMELY IMPORTANT that you first think about the task and the best way to solve it.
- IT IS EXTREMELY IMPORTANT that you break the task into smaller, simpler and more manageable subtasks.
- IT IS EXTREMELY IMPORTANT that you create a plan file called PLAN.md that contains the detailed plan for the task.
- IT IS EXTREMELY IMPORTANT that you do not add unnecessary changes or complexity in your plan.
- IT IS EXTREMELY IMPORTANT that you use the plan to track your progress. Make sure to update the plan as you complete each subtask.
- IT IS EXTREMELY IMPORTANT that you return your response as JSON with 'reasoning', 'tool_calls', and 'confidence' fields.
    - Here is an example of the expected format:
    ```json
    {{
        "reasoning": "I need to create a plan to solve the task.",
        "tool_calls": [
            {{
                "name": "edit_file",
                "arguments": {{
                    "file_path": "PLAN.md",
                    "content": "Detailed plan for the task."
                }}
            }}
        ],
        "confidence": 1.0
    }}
    ```

# Context
{self.context.model_dump_json(indent=2)}
"""


class PlannerAgentTool(Tool):
    name: str = "planner_agent"

    def run(
        self, agent_config: AgentConfig, researcher_output: ResearchOutput
    ) -> PlannerOutput:
        context = PlannerContext(researcher_output=researcher_output)
        agent = PlannerAgent(config=agent_config, context=context)
        agent.start()
        return agent.get_output()

    def get_schema(self) -> str | Callable:
        def planner_agent(task_description: str) -> PlannerOutput:
            """
            Use this tool to use an agent to plan how to solve the task.

            Args:
                task_description: A description that summarizes the task to solve.
            Returns:
                PlannerOutput: The output of the planner agent.
            """
            pass

        return planner_agent
