from typing import Callable
from agents.agent import (
    Agent,
    AgentConfig,
    AgentOutput,
    Context,
    LLMResponse,
    ResearchOutput,
)
from agents.tools import (
    Error,
    File,
    SearchWebToolResult,
    Tool,
    ReadFileToolResult,
    Ticket,
    GetTicketToolResult,
)


class ResearchContext(Context):
    ticket: Ticket | None = None
    files: list[File] = []
    web_search_results: list[SearchWebToolResult] = []


class ResearchAgent(Agent):
    name: str = "researcher_agent"
    context: ResearchContext = ResearchContext()

    def is_done(self):
        if self.context.num_iterations >= self.config.max_iterations:
            return True
        # TODO: figure out a way to determine whether agent has read all relevant files
        #  available in the repository.
        # We use 5 as a minimum number of iterations to ensure that the agent
        # has had a chance to gather some information.
        if (
            self.context.ticket
            and self.context.files
            and self.context.num_iterations >= 5
        ):
            return True
        return False

    def get_output(self) -> ResearchOutput:
        return ResearchOutput(
            ticket=self.context.ticket,
            files=self.context.files,
            web_search_results=self.context.web_search_results,
        )

    def step(self) -> tuple[LLMResponse | None, Error | None]:
        prompt = self.build_prompt(self.config.task_description)
        response, error = self._invoke_llm(prompt)
        self._update_common_context(response, error)

        if error is not None:
            return response, error

        for tool_call in response.tool_calls:
            if tool_call.name == "researcher_agent":
                # make sure that agent does not call itself recursively
                error = Error(
                    description="Research agent should not call itself recursively.",
                )
                return None, error
            if tool_call.name == "get_ticket":
                get_ticket_result = GetTicketToolResult(**tool_call.response)
                self.context.ticket = get_ticket_result.ticket
            if tool_call.name == "read_file":
                read_file_result = ReadFileToolResult(**tool_call.response)
                file = File(
                    path=read_file_result.file_path,
                    content=read_file_result.content,
                )
                self.context.files.append(file)

            if tool_call.name == "search_web":
                search_result = SearchWebToolResult(**tool_call.response)
                self.context.web_search_results.append(search_result)

        return response, None

    def validate_llm_response(self, llm_response: LLMResponse) -> Error | None:
        error = super().validate_llm_response(llm_response)

        # try to nudge the agent to use a tool if it decides to not use any tools
        if not llm_response.tool_calls:
            tools_to_use = []
            if not self.context.ticket:
                tools_to_use.append("get_ticket")
            if not self.context.files:
                tools_to_use += ["find_files", "read_file"]
            if not self.context.web_search_results:
                tools_to_use.append("search_web")

            error = Error(
                description=f"You decided not to use any tools. Please use the available tools at least once to gather information.",
                data={"available_tools": tools_to_use},
            )
        return error

    def build_prompt(self, task_description: str) -> str:
        return f"""
# Task
You are a software engineer that specializes in researching and gathering information to solve a problem or task.

Here is the task description:
{task_description}

Make sure to use the following tools to gather information:
- get_ticket: to get additional context about the task from a ticketing system
- read_file: to read files in the repository that might be relevant to the task
- search_web: to search the web for additional information that might be relevant to the task

# Guidelines
- IT IS EXTREMELY IMPORTANT that you return your response as JSON with 'reasoning', 'tool_calls', and 'confidence' fields.
    - Here is an example of the expected format:
    ```json
    {{
        "reasoning": "I need to read the files in the repository and search the web for additional information to solve the task.",
        "tool_calls": [
            {{
                "name": "get_ticket",
                "arguments": {{
                    "ticket_id": "123456"
                }}
            }},
            {{
                "name": "read_file",
                "arguments": {{
                    "file_path": "main.py"
                }}
            }},
            {{
                "name": "search_web",
                "arguments": {{
                    "query": "bug ticket description"
                }}
            }}
        ],
        "confidence": 1.0
    }}  
    ```

# Context
{self.context.model_dump_json(indent=2)}
"""


class ResearchAgentTool(Tool):
    name: str = "researcher_agent"

    def run(self, agent_config: AgentConfig) -> ResearchOutput:
        agent = ResearchAgent(config=agent_config)
        agent.start()
        return agent.get_output()

    def get_schema(self) -> str | Callable:
        def researcher_agent(task_description: str) -> ResearchOutput:
            """
            Use this tool to use an agent to research the task and gather more information.

            Args:
                task_description: A description that summarizes the task to solve.
            Returns:
                ResearchOutput: The output of the research agent.
            """
            pass

        return researcher_agent
