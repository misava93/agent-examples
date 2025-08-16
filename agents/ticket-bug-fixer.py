import difflib
import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

import ollama
from pydantic import BaseModel


class Ticket(BaseModel):
    id: str
    title: str = ""
    description: str = ""
    repository: str = ""


class File(BaseModel):
    path: str = ""
    content: str = ""
    includes_tests: bool = False


class WebSearchResult(BaseModel):
    query: str = ""
    results: List[str] = []


class CodeChange(BaseModel):
    description: str = ""
    git_patch: str = ""


class CodeChangeError(CodeChange):
    error: str = ""


Error = str


class ToolCall(BaseModel):
    name: str = ""
    arguments: Dict[str, Any] = {}
    response: Dict[str, Any] = {}

    def __hash__(self):
        return hash((self.name, frozenset(self.arguments.items())))


class ToolFunction(BaseModel):
    function: ToolCall


class LLMResponse(BaseModel):
    reasoning: str = ""
    tool_calls: List[ToolCall] = []
    confidence: float = 0.0


class Context(BaseModel):
    num_iterations: int = 0
    ticket: Ticket
    files_found: list[str] = []
    files: list[File] = []
    web_search_results: list[WebSearchResult] = []
    code_change_history: List[CodeChange] = []
    code_change_errors: List[CodeChangeError] = []
    tool_usage_history: List[ToolCall] = []
    is_bug_fixed: bool = False
    wrote_tests: bool = False
    # Note: we could model this with a richer structure. For now, we keep it simple.
    errors: list[Error] = []
    last_llm_thought: str = ""


class Prompt(BaseModel):
    instruction: str = """
# Overview
You are a software engineer that specializes in fixing bugs. Your task is to analyze the bug report and fix the underlying issue. Use the available tools to:
1. Process the ticket
2. Gather more context if needed
3. Identify the root cause
4. Write test to reproduce the bug
5. Implement a fix
6. Validate that the fix works

# Guidelines   
- IT IS EXTREMELY IMPORTANT that you return your response as JSON with 'reasoning', 'tool_calls', and 'confidence' fields.
- IT IS EXTREMELY IMPORTANT that you write tests to reproduce the bug after you have identified the root cause and before you implement the fix.
- IT IS EXTREMELY IMPORTANT that you first understand the relevant source code before implementing the fix.
- IT IS EXTREMELY IMPORTANT that you do not add unnecessary changes or complexity.
    - For example, do not create new files or directories unless necessary.
"""
    context: Context
    available_tools: List[str]

    def build_prompt(self) -> str:
        return self.model_dump_json(indent=2)

    def build_prompt_2(self) -> str:
        return f"""
{self.instruction}

# Context
## Ticket
{self.context.ticket.model_dump_json()}

## Files
{"\n".join([f.model_dump_json() for f in self.context.files])}

## Web Search Results
{"\n".join([r.model_dump_json() for r in self.context.web_search_results])}

## Tool Call History
{"\n".join([t.model_dump_json() for t in self.context.tool_usage_history])}

## Last LLM Thought
{self.context.last_llm_thought}

## Previous Errors
{"\n".join(self.context.errors)}

## Validation Context
{{"is_bug_fixed": {self.context.is_bug_fixed}, "wrote_tests": {self.context.wrote_tests}}}
"""


class BugFixResponse(BaseModel):
    success: bool
    diff: Optional[str] = None
    tests: Optional[str] = None
    documentation: Optional[str] = None
    error_message: Optional[str] = None


def get_ticket(ticket_id: str) -> Ticket:
    """
    Retrieves the ticket with the bug report

    You can use this to gather more information about the bug report.

    Args:
        ticket_id: The id of the ticket to retrieve
    Returns:
        Ticket: The ticket with the bug report
    """
    # This is a mock for now for simplicity’s sake:
    #   -  We use the current working directory as the repository
    # An actual implementation would retrieve the ticket from a database or
    # ticketing system (GitHub, Jira, etc.)
    cwd = os.getcwd()
    return Ticket(
        id=ticket_id,
        title="Bug Report",
        description='When I run the application and try to access the health endpoint, I get the following error: "Internal Server Error".',
        repository=cwd,
    )


def find_files(directory: str) -> list[str]:
    """
    Finds all files in the specified directory

    Args:
        directory: The directory to search
    Returns:
        list[str]: List of file paths
    """
    result: list[str] = []
    for root, dirs, files in os.walk(directory):
        # Remove hidden directories from dirs list in-place
        # This prevents os.walk from descending into them
        dirs[:] = [d for d in dirs if not d.startswith(".") and not d == "__pycache__"]
        for file in files:
            # for simplicity’s sake, lets just include these files only
            if file in {"Makefile", ".python-version", "main.py"}:
                result.append(os.path.join(root, file))

    return result


def read_file(file_path: str) -> str:
    """
    Reads the contents of a specific file

    Args:
        file_path: The path to the file to read
    Returns:
        str: The contents of the file
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


def search_web(query: str) -> str:
    """
    Searches the web for information related to the bug description

    Args:
        query: The search query
    Returns:
        str: Search results as a string
    """
    return "Not implemented yet"


def edit_file(file_path: str, content: str, includes_tests: bool = False) -> File:
    """
    Edits a file with the provided content. It overwrites the file.

    You can use this tool to:
        - implement a fix for the bug
        - write and/or update tests
        - refactor code
        - update documentation

    Args:
        file_path: The path to the file to edit
        content: The new content to write to the file
        includes_tests: Whether the edited content includes new or modified tests
    Returns:
        File: The file that was edited
    """
    return File(
        path=file_path,
        content=content,
        includes_tests=includes_tests,
    )


def calculate_diff(file_1: File, file_2: File) -> str:
    """
    Calculates the diff between two files

    Args:
        file_1: The first file
        file_2: The second file
    Returns:
        str: The diff between the two files
    """

    diff_result = difflib.unified_diff(
        file_1.content.splitlines(keepends=True),
        file_2.content.splitlines(keepends=True),
        fromfile=file_1.path,
        tofile=file_2.path,
    )

    return "".join(diff_result)


def handle_file_edited(context: Context, file: File) -> Context:
    """
    Handles the file edited by the agent

    Args
        context: The current context
        file: The file that was edited
    Returns:
        Context: The updated context
    """
    diff = ""
    # only allow updating files inside the repository
    if not file.path.startswith(context.ticket.repository):
        context.errors.append(
            f"Attempted to edit file outside of repository.{{'file_path': '{file.path}', 'repository': '{context.ticket.repository}'}}"
        )
        return context

    path = Path(file.path)
    # do not allow editing files in disallowed list
    disallowed_files = {
        "Makefile",
        ".python-version",
        "ticket-bug-fixer.py",
        "requirements.txt",
        ".gitignore",
    }
    if path.name in disallowed_files:
        context.errors.append(
            f"Attempted to edit disallowed file.{{'file_path': '{file.path}'}}"
        )
        return context

    found_file = False
    for f in context.files:
        if f.path == file.path:
            diff = calculate_diff(f, file)
            f.content = file.content
            found_file = True

    if not found_file:
        diff = calculate_diff(File(path=file.path, content=""), file)
        context.files.append(file)

    # save/update the file on disk
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(file.content, encoding="utf-8")

    context.tool_usage_history.append(
        ToolCall(
            name="edit_file",
            arguments={"file_path": file.path, "diff": diff},
            response={"success": True},
        )
    )
    if file.includes_tests:
        context.wrote_tests = True

    return context


class SubprocessResult(BaseModel):
    success: bool
    return_code: int
    stdout: str
    stderr: str
    command: str


def run_make_target(
    target, makefile_dir=None, makefile_name="Makefile"
) -> SubprocessResult:
    """
    Run a specific make target and return results

    Args:
        target: Make target to run (e.g., 'build', 'test', 'clean')
        makefile_dir: Directory containing the Makefile (defaults to current dir)
        makefile_name: Name of the makefile (defaults to 'Makefile')
    """
    # Change to makefile directory if specified
    original_dir = os.getcwd()
    if makefile_dir:
        os.chdir(makefile_dir)

    cmd = ["make", target]
    if makefile_name != "Makefile":
        cmd = ["make", "-f", makefile_name, target]

    # add PYTHONPATH to the environment of the subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = makefile_dir

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            env=env,
        )

        return SubprocessResult(
            success=result.returncode == 0,
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            command=" ".join(cmd),
        )

    except subprocess.TimeoutExpired:
        return SubprocessResult(
            success=False,
            return_code=-1,
            stdout="",
            stderr="Command timed out",
            command=" ".join(cmd),
        )

    except FileNotFoundError:
        return SubprocessResult(
            success=False,
            return_code=-1,
            stdout="",
            stderr="make command not found. Please install make.",
            command=" ".join(cmd),
        )
    except Exception as e:
        return SubprocessResult(
            success=False,
            return_code=-1,
            stdout="",
            stderr=f"An unexpected error occurred: {str(e)}",
            command=" ".join(cmd),
        )

    finally:
        # Restore original directory
        os.chdir(original_dir)


class ValidationError(BaseModel):
    description: str = ""
    data: Any = None


def validate_fix() -> Error | None:
    """
    Validates tha the bug has been fixed

    Returns:
        Error: The error if the validation failed, None otherwise
    """
    # NOTE: we have a separate handler for the tool call to avoid having the
    # LLM to pass the entire context to the tool.
    return None


def handle_validate_fix(context: Context) -> tuple[Context, ValidationError | None]:
    validation_error = None
    try:
        # step 1: check that tests were written
        if not context.wrote_tests:
            validation_error = ValidationError(
                description="You still have not written any tests. Please write tests before validating the fix again.",
                data=None,
            )
            return context, validation_error
        # step 2: run tests
        validation_error = test_code(context.ticket.repository)
        if validation_error is not None:
            return context, validation_error
        # step 3: check lint
        validation_error = lint_code(context.ticket.repository)
        if validation_error is not None:
            return context, validation_error
        # step 4: check security (TODO)
    except Exception as e:
        validation_error = ValidationError(
            description="An unexpected error occurred while trying to validate the fix.",
            data=str(e),
        )
        return context, validation_error
    finally:
        if validation_error is None:
            context.is_bug_fixed = True
        else:
            context.errors.append(validation_error.model_dump_json())

        # always update context with tool usage
        tool_usage = ToolCall(
            name="validate_fix",
            arguments={},
            response={
                "success": context.is_bug_fixed,
                "error": (
                    validation_error.model_dump()
                    if validation_error is not None
                    else None
                ),
            },
        )
        context.tool_usage_history.append(tool_usage)

        # always return update context and error (if any)
        return context, validation_error


def test_code(root_dir: str) -> ValidationError | None:
    """
    Runs the tests

    Args:
        root_dir: The root directory of the project

    Returns:
        ValidationError: The error if the tests failed, None otherwise
    """
    make_result = run_make_target("test", makefile_dir=root_dir)
    if not make_result.success:
        return ValidationError(
            description="Tests failed. Please fix the tests before validating the fix again.",
            data=make_result.model_dump(),
        )
    return None


def lint_code(root_dir: str) -> ValidationError | None:
    """
    Runs the lint check on the source code

    Args:
        root_dir: The root directory of the project

    Returns:
        ValidationError: The error if the lint check failed, None otherwise
    """
    make_result = run_make_target("lint-check", makefile_dir=root_dir)
    if not make_result.success:
        return ValidationError(
            description="Lint check failed. Please fix the lint errors before validating the fix again.",
            data=make_result.model_dump(),
        )
    return None


def format_code(root_dir: str) -> ValidationError | None:
    """
    Formats the source code to follow linting standards

    You can use this tool to try to fix lint check errors.

    Args:
        root_dir: The root directory of the project

    Returns:
        ValidationError: The error if the lint formatting failed, None otherwise
    """
    make_result = run_make_target("lint-fmt", makefile_dir=root_dir)
    if not make_result.success:
        return ValidationError(
            description="Lint formatting failed. Please fix the lint errors yourself before validating the fix again.",
            data=make_result.model_dump(),
        )
    return None


def parse_llm_response(
    response, context: Context
) -> tuple[tuple[Context, LLMResponse], ValidationError | None]:
    """
    Parses the LLM response into a structured format
    Args:
        response: The raw response from the LLM
        context: The current context
    Returns:
        tuple[Context, list[ToolCall]]: The updated context and the parsed tool calls
        if parsing was successful, ValidationError otherwise.
    """
    # NOTE: the response from Ollama SDK does not always include tool calls in the structured response.
    # When this happens, the tool call JSON definition is usually present in the raw
    # response/message from the LLM after the thinking portion.
    # This is a fallback mechanism to extract the tool calls from the raw response.
    validation_error = None
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
        validation_error = ValidationError(
            description="Failed to parse LLM response.",
            data=str(e),
        )
        # update context with error
        context.errors.append(validation_error.model_dump_json())
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
        # step 4: update context
        context.last_llm_thought = llm_response.reasoning
        return (context, llm_response), validation_error


def validate_llm_response(
    context: Context, llm_response: LLMResponse
) -> tuple[Context, ValidationError | None]:
    """
    Validates the LLM response
    Args:
        context: The current context
        llm_response: The raw response from the LLM
    Returns:
        ValidationError: The error if the response is invalid, None otherwise
    """
    validation_error = None
    # step 1: check that the LLM is not stuck repeating the same tool calls over and over again.
    total_tool_calls = context.tool_usage_history + llm_response.tool_calls
    if len(total_tool_calls) >= 3:
        n = 3
        last_n_tool_calls = context.tool_usage_history[-n:]
        # if all are equal
        are_all_equal = len(set(last_n_tool_calls)) == 1
        tool_name = last_n_tool_calls[0].name
        if are_all_equal and tool_name in {"get_ticket", "find_files"}:
            validation_error = ValidationError(
                description="LLM is stuck repeating the same tool calls over and over again.",
                data={
                    "tool_name": tool_name,
                },
            )
            context.errors.append(validation_error.model_dump_json())
    # step 2: security checks (e.g. check against common security vulnerabilities, etc)
    # step 3: apply content moderation policies
    return context, validation_error


def run_bug_fix_agent(max_iterations: int = 5):
    """
    Main agent loop that iteratively attempts to fix bugs
    Args:
        max_iterations: Maximum number of iterations to attempt
    """
    print("Please provide ticket ID with bug report (any string is fine): ")
    ticket_id = input()
    ticket_id = ticket_id.strip()
    iteration = 0
    context = Context(
        ticket=Ticket(id=ticket_id),
    )
    # prompt = Prompt(
    #     task=task,
    #     available_tools=[
    #         'get_ticket', 'find_files', 'read_file', 'search_web',
    #         'code_change', 'validate_fix'
    #     ],
    # )
    max_iterations = 25
    while True:
        # terminal condition 1: bug is fixed
        if context.is_bug_fixed:
            print("Bug has been fixed!")
            break
        if iteration >= max_iterations:
            print("Max iterations reached. Stopping early.")
            break

        prompt = Prompt(
            context=context,
            available_tools=[
                "get_ticket",
                "find_files",
                "read_file",
                "search_web",
                "edit_file",
                "validate_fix",
                "format_code",
            ],
        )
        prompt_str = prompt.build_prompt()
        try:
            response = ollama.chat(
                "gpt-oss:20b",
                messages=[{"role": "user", "content": prompt_str}],
                tools=[
                    get_ticket,
                    find_files,
                    read_file,
                    search_web,
                    edit_file,
                    validate_fix,
                    format_code,
                ],
            )
        except Exception as e:
            print(f"Error invoking LLM via Ollama client: {e}")
            continue

        (context, llm_response), validation_error = parse_llm_response(
            response, context
        )
        if validation_error is not None:
            # try again
            iteration += 1
            continue

        context, validation_error = validate_llm_response(context, llm_response)
        if validation_error is not None:
            # try again
            iteration += 1
            continue

        print("----------------------------------------------")
        print(f"Iteration {iteration}")
        print(f"Prompt:\n{prompt_str}")
        print(f"LLM Response Thinking:\n{llm_response.reasoning}")
        print(f"LLM Response Tool Calls:\n{llm_response.tool_calls}")
        print(f"LLM Response Confidence:\n{llm_response.confidence}")
        print("----------------------------------------------")

        if not llm_response.tool_calls:
            context.errors += [
                "You decided to not use any tools, but the bug fix has not been validated. Use the available tools to fix the bug."
            ]
            iteration += 1
            continue

        # Process the response to call the function if a tool call is present
        tool_functions = [ToolFunction(function=tc) for tc in llm_response.tool_calls]
        for tool in tool_functions:
            match tool.function.name:
                case "get_ticket":
                    ticket = get_ticket(
                        ticket_id=tool.function.arguments["ticket_id"],
                    )
                    context.ticket = ticket
                    tool_call = ToolCall(
                        name=tool.function.name,
                        arguments=tool.function.arguments,
                        response={"ticket": ticket.model_dump()},
                    )
                    context.tool_usage_history.append(tool_call)
                case "find_files":
                    if context.ticket.repository == "":
                        context.errors.append(
                            "You must retrieve the ticket before finding files."
                        )
                        break
                    # check that lower priority tasks are completed before
                    # higher priority tasks
                    file_paths = find_files(
                        directory=context.ticket.repository,
                    )
                    context.files_found = list(
                        set(context.files_found).union(set(file_paths))
                    )
                    tool_call = ToolCall(
                        name=tool.function.name,
                        arguments=tool.function.arguments,
                        response={"files": [f for f in file_paths]},
                    )
                    context.tool_usage_history.append(tool_call)
                case "read_file":
                    file_path = tool.function.arguments["file_path"]
                    contents = read_file(
                        file_path=file_path,
                    )
                    for f in context.files:
                        if f.path == file_path:
                            f.content = contents
                    tool_call = ToolCall(
                        name=tool.function.name,
                        arguments=tool.function.arguments,
                        response={"contents": contents},
                    )
                    context.tool_usage_history.append(tool_call)
                case "search_web":
                    result_raw = search_web(
                        query=tool.function.arguments["query"],
                    )
                    web_search_result = WebSearchResult(
                        query=tool.function.arguments["query"],
                        results=[result_raw],
                    )
                    context.web_search_results.append(web_search_result)
                    tool_call = ToolCall(
                        name=tool.function.name,
                        arguments=tool.function.arguments,
                        response={"results": [result_raw]},
                    )
                    context.tool_usage_history.append(tool_call)
                case "edit_file":
                    updated_file = edit_file(
                        file_path=tool.function.arguments["file_path"],
                        content=tool.function.arguments["content"],
                        includes_tests=tool.function.arguments.get(
                            "includes_tests", False
                        ),
                    )
                    context = handle_file_edited(context, updated_file)
                case "validate_fix":
                    context, validation_error = handle_validate_fix(context)
                case "format_code":
                    # Note: we ignore the arguments provided by the LLM. This
                    # is to improve correctness as given the current implementation,
                    # we know that we want to run checks from the root of the
                    # repository.
                    validation_error = format_code(context.ticket.repository)
                    if validation_error is not None:
                        context.errors.append(validation_error.model_dump_json())
                    tool_call = ToolCall(
                        name=tool.function.name,
                        arguments=tool.function.arguments,
                        response={
                            "success": validation_error is None,
                            "error": (
                                validation_error.model_dump()
                                if validation_error is not None
                                else None
                            ),
                        },
                    )
                    context.tool_usage_history.append(tool_call)
                case _:
                    print(f"Unknown function: {tool.function.name}")
        iteration += 1
        context.num_iterations = iteration

    print(f"Final context: {context.model_dump_json(indent=2)}")


if __name__ == "__main__":
    run_bug_fix_agent()
