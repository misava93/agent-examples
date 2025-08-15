import difflib
import enum
import json
import os
import urllib
from typing import List, Dict, Any, Optional

import git
import ollama
import requests
from pydantic import BaseModel
from bs4 import BeautifulSoup

class Ticket(BaseModel):
    id: str
    title: str = ""
    description: str = ""
    repository: str = ""

class File(BaseModel):
    path: str = ""
    content: str = ""

class WebSearchResult(BaseModel):
    query: str = ""
    results: List[str] = []

class CodeChange(BaseModel):
    description: str = ""
    git_patch: str = ""

class CodeChangeError(CodeChange):
    error: str = ""

class ToolUsage(BaseModel):
    tool_name: str = ""
    arguments: Dict[str, Any] = {}

class Context(BaseModel):
    num_iterations: int = 0
    ticket: Ticket
    files: list[File] = []
    web_search_results: list[WebSearchResult] = []
    code_change_history: List[CodeChange] = []
    code_change_errors: List[CodeChangeError] = []
    tool_usage_history: List[ToolUsage] = []
    is_bug_fixed: bool = False
    # Note: we could model this with a richer structure. For now, we keep it simple.
    errors: list[str] = []

class Prompt(BaseModel):
    instruction: str = """
Analyze the bug report and fix the bug. Use the available tools to:
1. Process the ticket
2. Gather more context if needed
3. Identify the root cause
4. Write test to reproduce the bug
5. Implement a fix
6. Validate the fix works
    
Return your response as JSON with 'reasoning', 'tool_calls', and 'confidence' fields.
"""
    context: Context
    available_tools: List[str]

    def build_prompt(self) -> str:
        return self.model_dump_json(indent=2)

class BugFixResponse(BaseModel):
    success: bool
    diff: Optional[str] = None
    tests: Optional[str] = None
    documentation: Optional[str] = None
    error_message: Optional[str] = None

def get_ticket(ticket_id: str) -> Ticket:
    """
    Retrieves ticket with bug report
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
        title='Bug Report',
        description='When I run the application and try to access the health endpoint, I get the following error: "Internal Server Error".',
        repository=cwd,
    )

def find_files(directory: str) -> list[str]:
    """
    Finds all files in the specified directory

    Note: It ignores hidden files and directories.

    Args:
        directory: The directory to search
    Returns:
        list[str]: List of file paths
    """
    result: list[str] = []
    for root, dirs, files in os.walk(directory):
        # Remove hidden directories from dirs list in-place
        # This prevents os.walk from descending into them
        dirs[:] = [d for d in dirs if not d.startswith('.') and not d == "__pycache__"]
        for file in files:
            # for simplicity’s sake, lets just include the api.py file for now
            if file == "api.py":
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
        with open(file_path, 'r', encoding='utf-8') as f:
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

def edit_file(file_path: str, content: str) -> File:
    """
    Edits a file with the provided content. It overwrites the file.
    Args:
        file_path: The path to the file to edit
        content: The new content to write to the file
    Returns:
        File: The file that was edited
    """
    return File(
        path=file_path,
        content=content,
    )

def calculate_diff(file_1: File, file_2: File, context_lines=3) -> str:
    """Generate unified diff between two files"""

    diff_result = difflib.unified_diff(
        file_1.content.splitlines(keepends=True),
        file_2.content.splitlines(keepends=True),
        fromfile=file_1.path,
        tofile=file_2.path,
        n=context_lines  # Number of context lines
    )

    return ''.join(diff_result)

def handle_file_edited(context: Context, file: File) -> Context:
    """
    Handles the file edited by the agent
    Args:
        context: The current context
        file: The file that was edited
    Returns:
        Context: The updated context
    """
    diff = ""
    for f in context.files:
        if f.path == file.path:
            diff = calculate_diff(f, file)
            f.content = file.content

    # save/update the file on disk
    with open(file.path, 'w', encoding='utf-8') as f:
        f.write(file.content)

    context.tool_usage_history.append(
        ToolUsage(
            tool_name="edit_file",
            arguments={"file_path": file.path, "diff": diff},
        )
    )

    return context


def validate_fix():
    pass


def run_bug_fix_agent(max_iterations: int = 5):
    """
    Main agent loop that iteratively attempts to fix bugs
    Args:
        max_iterations: Maximum number of iterations to attempt
    """
    print("Please provide ticket ID with bug report: ")
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
    done = False
    while not done:
        prompt = Prompt(
            context=context,
            available_tools=[
                'get_ticket', 'find_files', 'read_file', 'search_web',
                'edit_file', 'validate_fix'
            ],
        )
        prompt_str = prompt.build_prompt()
        try:
            response = ollama.chat(
                'qwen3:8b',
                messages=[{'role': 'user', 'content': prompt_str}],
                tools=[
                    get_ticket, find_files, read_file, search_web,
                    edit_file, validate_fix
                ],
            )
        except Exception as e:
            print(f"Error: {e}")
            continue

        print(response.message)

        if response.message.tool_calls is None:
            if not context.is_bug_fixed:
                context.errors += ["You decided to not use any tools, but the bug fix has not been validated."]
                iteration += 1
                continue
            else:
                # is done
                break
        # Process the response to call the function if a tool call is present
        for tool in response.message.tool_calls or []:
            match tool.function.name:
                case 'get_ticket':
                    ticket = get_ticket(**tool.function.arguments)
                    context.ticket = ticket
                    tool_usage = ToolUsage(
                        tool_name=tool.function.name,
                        arguments=tool.function.arguments,
                    )
                    context.tool_usage_history.append(tool_usage)
                case 'find_files':
                    # check that lower priority tasks are completed before
                    # higher priority tasks
                    file_paths = find_files(**tool.function.arguments)
                    files = [File(path=path) for path in file_paths]
                    context.files = files
                    tool_usage = ToolUsage(
                        tool_name=tool.function.name,
                        arguments=tool.function.arguments,
                    )
                    context.tool_usage_history.append(tool_usage)
                case 'read_file':
                    file_path = tool.function.arguments['file_path']
                    contents = read_file(**tool.function.arguments)
                    for f in context.files:
                        if f.path == file_path:
                            f.content = contents
                    tool_usage = ToolUsage(
                        tool_name=tool.function.name,
                        arguments=tool.function.arguments,
                    )
                    context.tool_usage_history.append(tool_usage)
                case 'search_web':
                    result_raw = search_web(**tool.function.arguments)
                    web_search_result = WebSearchResult(
                        query=tool.function.arguments['query'],
                        results=[result_raw],
                    )
                    context.web_search_results.append(web_search_result)
                    tool_usage = ToolUsage(
                        tool_name=tool.function.name,
                        arguments=tool.function.arguments,
                    )
                    context.tool_usage_history.append(tool_usage)
                case 'edit_file':
                    updated_file = edit_file(**tool.function.arguments)
                    context = handle_file_edited(context, updated_file)
                case _:
                    print(f"Unknown function: {tool.function.name}")
        iteration += 1
        context.num_iterations = iteration


if __name__ == "__main__":
    run_bug_fix_agent()
