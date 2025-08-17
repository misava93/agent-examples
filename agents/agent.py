from abc import ABC, abstractmethod
import logging
import re
from typing import List, Literal

from ollama import ChatResponse
import ollama
from pydantic import BaseModel
from agents.tools import EditFileTool, Error, File, FindFilesTool, FormatCodeTool, GetTicketTool, LintCodeTool, ReadFileTool, SearchWebTool, TestCodeTool, Tool, ToolCall, ToolFunction

class LLMResponse(BaseModel):
    reasoning: str = ""
    tool_calls: List[ToolCall] = []
    confidence: float = 0.0

class Context(BaseModel):
    num_iterations: int = 0
    errors: list[Error] = []
    last_llm_thought: str = ""
    tool_calls_history: list[ToolCall] = []

class AgentConfig(BaseModel):
    name: str
    model: str
    tools: dict[str, Tool]
    max_iterations: int = 10
    task_description: str
    root_directory: str = ""
    # files that are not allowed to be edited or read
    disallowed_files: list[str] = []

class AgentOutput(BaseModel):
    context: Context
    error: Error | None = None

def parse_llm_response(
    response: ChatResponse,
) -> tuple[LLMResponse | None, Error | None]:
    """
    Parses the LLM response into a structured format
    Args:
        response: The raw response from the LLM
    Returns:
        tuple[LLMResponse | None, Error | None]: The parsed LLM response or an error if parsing was unsuccessful.
    """
    # NOTE: the response from Ollama SDK does not always include tool calls in the structured response.
    # When this happens, the tool call JSON definition is usually present in the raw
    # response/message from the LLM after the thinking portion.
    # This is a fallback mechanism to extract the tool calls from the raw response.
    error = None
    tool_calls: list[ToolCall] = [
        ToolCall(name=tc.function.name, arguments=tc.function.arguments)
        for tc in response.message.tool_calls or []
    ]
    thinking = response.message.thinking or ""
    llm_response_parsed = None
    try:
        # step 1: lets try to parse any structured JSON from the response content
        # fallback: try to parse tool calls from response content
        tokens = response.message.content.split("</think>")
        if len(tokens) > 1:
            raw_content = tokens[-1].strip()
            try:
                llm_response_parsed = LLMResponse.model_validate_json(raw_content)
            except Exception as _:
                # ignore errors because it is not guaranteed that the response will be valid JSON all the time.
                pass
        # step 2: lets try to parse thinking/reasoning from  the response content.
        # use regex to find everything after <think> and before </think> from response content
        regex = r"<think>(.*?)</think>"
        match = re.search(regex, response.message.content, re.DOTALL)
        if match:
            thinking = thinking or match.group(1)
    except Exception as e:
        error = Error(
            description="Failed to parse LLM response.",
            data=str(e),
        )
        return None, error

    finally:
        llm_response = LLMResponse(
            reasoning=thinking,
            tool_calls=tool_calls,
            confidence=1.0,
        )
        # step 3: if tools are missing, default to tool calls parsed from the raw response
        if llm_response_parsed:
            if not llm_response.reasoning:
                llm_response.reasoning = llm_response_parsed.reasoning
            if not llm_response.tool_calls:
                llm_response.tool_calls = llm_response_parsed.tool_calls
            if not llm_response.confidence or llm_response.confidence < 0.0:
                llm_response.confidence = llm_response_parsed.confidence
        
        return llm_response, error


class Agent(ABC, BaseModel):
    name: str
    config: AgentConfig
    current_iteration: int = 0
    context: Context = Context()
    
    @abstractmethod
    def is_done(self):
        pass

    @abstractmethod
    def build_prompt(self) -> str:
        pass

    @abstractmethod
    def get_output(self) -> AgentOutput:
        pass

    @abstractmethod
    def step(self) -> tuple[LLMResponse, Error | None]:
        """
        Executes a single step of the agent.

        Returns:
            tuple[LLMResponse, Error | None]: The LLM response and an error if the step failed.
        """
        pass
    
    def _update_common_context(self, response: LLMResponse, error: Error | None):
        if error is not None:
            # update context
            self.context.errors.append(error)
        else:
            self.context.num_iterations += 1
            self.context.last_llm_thought = response.reasoning
            self.context.tool_calls_history.extend(response.tool_calls)
        
        self.context.num_iterations += 1
        return response, error
    
    def start(self) -> AgentOutput:
        while not self.is_done():
            self.step()
        return self.get_output()
    

    def _validate_llm_response(
        self, context: Context, llm_response: LLMResponse
    ) -> Error | None:
        """
        Validates the LLM response
        Args:
            context: The current context
            llm_response: The parsed LLM response
        Returns:
            Error: The error if the response failed validation, None otherwise
        """
        # step 1: check that the LLM is not stuck repeating the same tool calls over and over again.
        total_tool_calls = context.tool_calls_history + llm_response.tool_calls
        if len(total_tool_calls) >= 3:
            n = 5
            last_n_tool_calls = context.tool_calls_history[-n:]
            # if all are equal
            are_all_equal = len(set(last_n_tool_calls)) == 1
            if are_all_equal:
                error = Error(
                    description="LLM is stuck repeating the same tool calls over and over again.",
                    data=last_n_tool_calls[0].model_dump_json(),
                )
                return error
        # step 2: TODO: security checks (e.g. check against common security vulnerabilities, etc)
        # step 3: TODO: apply content moderation policies
        # step 4:check to nudge LLM to use tools if needed
        if not llm_response.tool_calls:
            error = Error(
                description="LLM decided to not use any tools, but the task has not been completed yet. Please use the available tools to complete the task.",
            )
            return error
        return None

    def _invoke_llm(self, prompt: str) -> tuple[LLMResponse | None, Error | None]:
        try:
            response = ollama.chat(
                self.config.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[
                    tool.get_schema() for tool in self.config.tools.values()
                ],
            )
            llm_response, error = parse_llm_response(response)
            if error is not None:
                return None, error

            print("----------------------------------------------")
            print(f"Model: {self.config.model}")
            print(f"Agent name: {self.name}")
            print(f"Iteration {self.context.num_iterations}")
            print(f"Prompt:\n{prompt}")
            print(f"LLM Response Thinking:\n{llm_response.reasoning}")
            print(f"LLM Response Tool Calls:\n{llm_response.tool_calls}")
            print(f"LLM Response Confidence:\n{llm_response.confidence}")
            print(f"Context:\n{self.context.model_dump_json(indent=2)}")
            print("----------------------------------------------")

            error = self._validate_llm_response(self.context, llm_response)
            if error is not None:
                return None, error
            tool_calls, error = self._process_tool_calls(llm_response)
            if error is not None:
                return None, error
            # update the LLM response with the tool calls that are augmented with results
            llm_response.tool_calls = tool_calls
            return llm_response, None
        except Exception as e:
            error = Error(
                description="Failed to invoke LLM via Ollama client.",
                data=str(e),
            )
            return None, error

    
    def _process_tool_calls(self, response: LLMResponse) -> tuple[list[ToolCall], Error | None]:
        from agents.researcher import ResearchAgentTool
        from agents.planner import PlannerAgentTool
        from agents.coder import CoderAgentTool

        tool_calls = []
        # Process the response to call the function if a tool call is present
        tool_functions = [ToolFunction(function=tc) for tc in response.tool_calls]
        for tool_f in tool_functions:
            try:
                match tool_f.function.name:
                    case "get_ticket":
                        get_ticket = GetTicketTool()
                        tool_call_result = get_ticket.run(ticket_id=tool_f.function.arguments["ticket_id"])
                        tool_call = ToolCall(
                            name=tool_f.function.name,
                            arguments=tool_f.function.arguments,
                            response=tool_call_result.model_dump(),
                        )
                        tool_calls.append(tool_call)
                    case "find_files":
                        find_files = FindFilesTool()
                        directory = self.config.root_directory
                        if "directory" in tool_f.function.arguments:
                            directory = tool_f.function.arguments["directory"]
                        allowed_dirs = [self.config.root_directory]
                        tool_call_result = find_files.run(directory=directory, allowed_dirs=allowed_dirs)
                        file_paths = tool_call_result.file_paths
                        tool_call = ToolCall(
                            name=tool_f.function.name,
                            arguments=tool_f.function.arguments,
                            response={"files": [f for f in file_paths]},
                        )
                        tool_calls.append(tool_call)
                    case "read_file":
                        read_file = ReadFileTool()
                        tool_call_result = read_file.run(file_path=tool_f.function.arguments["file_path"])
                        tool_call = ToolCall(
                            name=tool_f.function.name,
                            arguments=tool_f.function.arguments,
                            response=tool_call_result.model_dump(),
                        )
                        tool_calls.append(tool_call)
                    case "search_web":
                        search_web = SearchWebTool()
                        tool_call_result = search_web.run(query=tool_f.function.arguments["query"])
                        tool_call = ToolCall(
                            name=tool_f.function.name,
                            arguments=tool_f.function.arguments,
                            response=tool_call_result.model_dump(),
                        )
                        tool_calls.append(tool_call)
                    case "edit_file":
                        read_file = ReadFileTool()
                        tool_call_result = read_file.run(file_path=tool_f.function.arguments["file_path"])
                        current_file = File(
                            path=tool_f.function.arguments["file_path"],
                            content=tool_call_result.content,
                        )

                        edit_file = EditFileTool()
                        tool_call_result = edit_file.run(
                            file_path=tool_f.function.arguments["file_path"],
                            content=tool_f.function.arguments["content"],
                            current_file=current_file,
                            allowed_dirs=[self.config.root_directory],
                            disallowed_files=self.config.disallowed_files,
                        )

                        tool_call = ToolCall(
                            name=tool_f.function.name,
                            arguments=tool_f.function.arguments,
                            response=tool_call_result.model_dump(),
                        )
                        tool_calls.append(tool_call)
                    case "test_code":
                        test_code = TestCodeTool()
                        root_dir = self.config.root_directory
                        if "root_dir" in tool_f.function.arguments:
                            root_dir = tool_f.function.arguments["root_dir"]
                        tool_call_result = test_code.run(root_dir=root_dir)
                        tool_call = ToolCall(
                            name=tool_f.function.name,
                            arguments=tool_f.function.arguments,
                            response=tool_call_result.model_dump(),
                        )
                        tool_calls.append(tool_call)
                    case "lint_code":
                        lint_code = LintCodeTool()
                        root_dir = self.config.root_directory
                        if "root_dir" in tool_f.function.arguments:
                            root_dir = tool_f.function.arguments["root_dir"]
                        tool_call_result = lint_code.run(root_dir=root_dir)
                        tool_call = ToolCall(
                            name=tool_f.function.name,
                            arguments=tool_f.function.arguments,
                            response=tool_call_result.model_dump(),
                        )
                        tool_calls.append(tool_call)
                    case "format_code":
                        format_code = FormatCodeTool()
                        root_dir = self.config.root_directory
                        if "root_dir" in tool_f.function.arguments:
                            root_dir = tool_f.function.arguments["root_dir"]
                        tool_call_result = format_code.run(root_dir=root_dir)
                        tool_call = ToolCall(
                            name=tool_f.function.name,
                            arguments=tool_f.function.arguments,
                            response=tool_call_result.model_dump(),
                        )
                        tool_calls.append(tool_call)
                    case "researcher_agent":
                        research_agent = ResearchAgentTool()
                        agent_config = self.config.model_copy()
                        # hack to restrict the tools to the ones that are needed for the research agent
                        agent_config.tools={
                            "get_ticket": GetTicketTool(),
                            "find_files": FindFilesTool(),
                            "read_file": ReadFileTool(),
                            "search_web": SearchWebTool(),
                        }
                        agent_output = research_agent.run(agent_config=agent_config)
                        tool_call = ToolCall(
                            name=tool_f.function.name,
                            arguments=tool_f.function.arguments,
                            response=agent_output.model_dump(),
                        )
                        tool_calls.append(tool_call)
                    case "planner_agent":
                        planner_agent = PlannerAgentTool()
                        agent_config = self.config.model_copy()
                        # hack to restrict the tools to the ones that are needed for the planner agent
                        agent_config.tools={
                            "read_file": ReadFileTool(),
                            "edit_file": EditFileTool(),
                        }
                        agent_output = planner_agent.run(agent_config=agent_config)
                        tool_call = ToolCall(
                            name=tool_f.function.name,
                            arguments=tool_f.function.arguments,
                            response=agent_output.model_dump(),
                        )
                        tool_calls.append(tool_call)
                    case "coder_agent":
                        coder_agent = CoderAgentTool()
                        agent_config = self.config.model_copy()
                        # hack to restrict the tools to the ones that are needed for the coder agent
                        agent_config.tools={
                            "find_files": FindFilesTool(),
                            "read_file": ReadFileTool(),
                            "edit_file": EditFileTool(),
                            "test_code": TestCodeTool(),
                            "lint_code": LintCodeTool(),
                            "format_code": FormatCodeTool(),
                        }
                        agent_output = coder_agent.run(agent_config=agent_config)
                        tool_call = ToolCall(
                            name=tool_f.function.name,
                            arguments=tool_f.function.arguments,
                            response=agent_output.model_dump(),
                        )
                        tool_calls.append(tool_call)
                    case _:
                        error = Error(
                            description="Unknown tool call.",
                            data={
                                "tool_call": tool_f.function.model_dump(),
                            },
                        )
                        return [], error
            except Exception as e:
                error = Error(
                    description="Failed to process tool calls.",
                    data={
                        "tool_call": tool_f.function.model_dump(),
                        "error": str(e),
                    },
                )
                return [], error
            
        return tool_calls, None
