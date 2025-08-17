from typing import Callable
from agents.agent import Agent, AgentConfig, AgentOutput, Context, LLMResponse
from pydantic import BaseModel

from agents.researcher import ResearchOutput
from agents.tools import Error, Tool


class PlannerContext(Context):
    researcher_output: ResearchOutput | None = None
    plan_file: str = ""
    plan_file_content: str = ""

class PlannerOutput(AgentOutput):
    plan_file_path: str

class PlannerAgent(Agent):
    name: str = "planner_agent"
    specific_context: PlannerContext = PlannerContext()
    
    def is_done(self):
        if self.context.num_iterations >= self.config.max_iterations:
            return True
        if self.specific_context.plan_file and self.specific_context.plan_file_content:
            return True
        return False

    def get_output(self) -> PlannerOutput:
        return PlannerOutput(
            plan_file_path=self.specific_context.plan_file,
        )

    def step(self) -> tuple[LLMResponse, Error | None]:
        prompt = self.build_prompt(self.config.task_description)
        response, error = self._invoke_llm(prompt)
        self._update_common_context(response, error)

        if error is not None:
            return response, error
        
        # check if an edit_file tool call was made with the file_path "PLAN.md"
        # and update the context if found
        for tool_call in response.tool_calls:
            if tool_call.name == "edit_file" and tool_call.arguments["file_path"] == "PLAN.md":
                self.specific_context.plan_file = tool_call.arguments["file_path"]
                self.specific_context.plan_file_content = tool_call.arguments["content"]
                break
        
        return response, error

    

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

# Common Context
{self.context.model_dump_json()}

# Specific Context
{self.specific_context.model_dump_json()}
"""

class PlannerAgentTool(Tool):
    name: str = "planner_agent"

    def run(self, agent_config: AgentConfig) -> PlannerOutput:
        agent = PlannerAgent(config=agent_config)
        agent.start()
        return agent.get_output()

    def get_schema(self) -> str|Callable:
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
   