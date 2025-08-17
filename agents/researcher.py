from typing import Callable
from agents.agent import Agent, AgentConfig, AgentOutput, Context, LLMResponse
from agents.tools import Error, File, SearchWebToolResult, Tool


class ResearchContext(Context):
    files: list[File] = []
    web_search_results: list[SearchWebToolResult] = []

class ResearchOutput(AgentOutput):
    files: list[File]
    web_search_results: list[str]

class ResearchAgent(Agent):
    name: str = "researcher_agent"
    specific_context: ResearchContext = ResearchContext()
    
    def is_done(self):
        # TODO: figure out a better way to determine if the agent is done
        if self.context.num_iterations >= self.config.max_iterations:
            return True
        return False

    def get_output(self) -> ResearchOutput:
        return ResearchOutput(
            files=self.specific_context.files,
            web_search_results=self.specific_context.web_search_results,
        )

    def step(self) -> tuple[LLMResponse, Error | None]:
        prompt = self.build_prompt(self.config.task_description)
        response, error = self._invoke_llm(prompt)
        self._update_common_context(response, error)
    
        if error is not None:
            return response, error
        
        # check if a read_file tool call was made
        # and update the context if found
        for tool_call in response.tool_calls:
            if tool_call.name == "read_file":
                file = File(
                    path=tool_call.arguments["file_path"],
                    content=tool_call.arguments["content"],
                )
                self.specific_context.files.append(file)
        
        # check if a search_web tool call was made
        # and update the context if found
        for tool_call in response.tool_calls:
            if tool_call.name == "search_web":
                search_result = SearchWebToolResult(
                    results=tool_call.arguments["results"],
                    query=tool_call.arguments["query"],
                )
                self.specific_context.web_search_results.append(search_result)
        
        return response, None

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

# Common Context
{self.context.model_dump_json()}

# Specific Context
{self.specific_context.model_dump_json()}
"""

class ResearchAgentTool(Tool):
    name: str = "researcher_agent"

    def run(self, agent_config: AgentConfig) -> ResearchOutput:
        agent = ResearchAgent(config=agent_config)
        agent.start()
        return agent.get_output()

    def get_schema(self) -> str|Callable:
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

   