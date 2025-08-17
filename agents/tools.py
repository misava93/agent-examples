from abc import ABC, abstractmethod
import os
import difflib
import subprocess
from pathlib import Path
from typing import Callable, Any, Dict
from pydantic import BaseModel


tools_available = [
    "find_files",
    "read_file",
    "search_web",
    "edit_file",
    "get_ticket",
    "test_code",
    "lint_code",
    "format_code",
]


class Error(BaseModel):
    description: str = ""
    data: Any = None


class ToolCall(BaseModel):
    name: str = ""
    arguments: Dict[str, Any] = {}
    response: Dict[str, Any] = {}

    def __hash__(self):
        return hash((self.name, frozenset(self.arguments.items())))


class ToolFunction(BaseModel):
    function: ToolCall


class ToolResult(BaseModel):
    error: Error | str | None = None


class Tool(ABC, BaseModel):
    name: str

    def get_name(self) -> str:
        return self.name

    @abstractmethod
    def run(self) -> ToolResult:
        pass

    @abstractmethod
    def get_schema(self) -> str | Callable:
        pass


class FindFilesToolResult(ToolResult):
    file_paths: list[str]


class FindFilesTool(Tool):
    name: str = "find_files"

    def run(self, directory: str, allowed_dirs: list[str]) -> FindFilesToolResult:
        result: list[str] = []
        for root, dirs, files in os.walk(directory):
            # Remove hidden directories from dirs list in-place
            # This prevents os.walk from descending into them
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and not d == "__pycache__"
                and d in allowed_dirs
            ]
            for file in files:
                result.append(os.path.join(root, file))

        return FindFilesToolResult(file_paths=result)

    def get_schema(self) -> str | Callable:
        # Note: Ollama supports dynamic schema generation for tools based
        # on function signatures.
        def find_files(directory: str) -> FindFilesToolResult:
            """
            Finds all files in the specified directory.

            This tool can be used to gather more context needed to solve a task.

            Args:
                directory: The directory to search
            Returns:
                FindFilesToolResult: List of file paths found in the directory
            """
            pass

        return find_files


class ReadFileToolResult(ToolResult):
    file_path: str = ""
    content: str = ""


class ReadFileTool(Tool):
    name: str = "read_file"

    def run(self, file_path: str) -> ReadFileToolResult:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                return ReadFileToolResult(file_path=file_path, content=content)
        except FileNotFoundError:
            return ReadFileToolResult(error=f"File not found. file_path={file_path}")
        except Exception as e:
            return ReadFileToolResult(
                error=f"Error reading file. file_path={file_path}, error={str(e)}"
            )

    def get_schema(self) -> str | Callable:
        def read_file(file_path: str) -> ReadFileToolResult:
            """
            Reads the contents of a specific file.

            This tool can be used to gather more context needed to solve a task.

            Args:
                file_path: The path to the file to read
            Returns:
                ReadFileToolResult: The contents of the file
            """
            pass

        return read_file


class SearchWebToolResult(ToolResult):
    query: str
    results: list[str] = []


class SearchWebTool(Tool):
    name: str = "search_web"

    def run(self, query: str) -> SearchWebToolResult:
        # This is a placeholder implementation
        # In a real implementation, this would use a web search API
        return SearchWebToolResult(query=query, results=["No search results found"])

    def get_schema(self) -> str | Callable:
        def search_web(query: str) -> SearchWebToolResult:
            """
            Searches the web for additional information related to the provided query.

            This tool can be used to gather more context needed to solve a task.

            Args:
                query: The search query
            Returns:
                SearchWebToolResult: Search results from the web for the provided query
            """
            pass

        return search_web


class File(BaseModel):
    path: str = ""
    content: str = ""
    diff: str = ""


class EditFileToolResult(ToolResult):
    file: File | None = None


class EditFileTool(Tool):
    name: str = "edit_file"

    def run(
        self,
        file_path: str,
        content: str,
        current_file: File,
        allowed_dirs: list[str],
        disallowed_files: list[str],
    ) -> EditFileToolResult:
        # only allow updating files inside the allowed directories
        any_allowed_dir = any(
            file_path.startswith(allowed_dir) for allowed_dir in allowed_dirs
        )
        if not any_allowed_dir:
            return EditFileToolResult(
                error=f"Attempted to edit file outside of allowed directories. Context={{'file_path': '{file_path}', 'allowed_dirs': '{allowed_dirs}'}}"
            )

        path = Path(file_path)
        # do not allow editing files in disallowed list
        disallowed_files = set(disallowed_files)
        if path.name in disallowed_files:
            return EditFileToolResult(
                error=f"Attempted to edit disallowed file. Context={{'file_path': '{file_path}'}}"
            )

        file = File(path=file_path, content=content)
        diff = self.calculate_diff(current_file, file)
        file.diff = diff

        # save/update the file on disk
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except Exception as e:
            return EditFileToolResult(
                error=f"Error saving file to disk. Context={{'file_path': '{file_path}', 'error': '{str(e)}'}}"
            )

        return EditFileToolResult(file=file)

    def get_schema(self) -> str | Callable:
        def edit_file(file_path: str, content: str) -> EditFileToolResult:
            """
            Edits a file with the provided content. It overwrites the file.

            You can use this tool to:
                - implement a code change
                - write and/or update tests
                - update documentation
                - etc.

            Args:
                file_path: The path to the file to edit
                content: The new content to write to the file
            Returns:
                EditFileToolResult: Result that indicates whether the file was successfully edited.
            """
            pass

        return edit_file

    def calculate_diff(self, file_1: File, file_2: File) -> str:
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


class Ticket(BaseModel):
    id: str
    title: str = ""
    description: str = ""
    repository: str = ""


class GetTicketToolResult(ToolResult):
    ticket: Ticket


class GetTicketTool(Tool):
    name: str = "get_ticket"

    def run(self, ticket_id: str) -> GetTicketToolResult:
        # This is a mock for now for simplicity’s sake:
        #   -  We use the current working directory as the repository
        # An actual implementation would retrieve the ticket from a database or
        # ticketing system (GitHub, Jira, etc.)
        cwd = os.getcwd()
        return GetTicketToolResult(
            ticket=Ticket(
                id=ticket_id,
                title="Bug Report",
                description='When I run the application and try to access the health endpoint, I get the following error: "Internal Server Error".',
                repository=cwd,
            )
        )

    def get_schema(self) -> str | Callable:
        def get_ticket(ticket_id: str) -> GetTicketToolResult:
            """
            Retrieves the ticket with the bug report

            You can use this to gather more information about a task.

            Args:
                ticket_id: The id of the ticket to retrieve
            Returns:
                GetTicketToolResult: The ticket with additional information about the task
            """
            pass

        return get_ticket


class SubprocessResult(BaseModel):
    success: bool
    return_code: int
    stdout: str
    stderr: str
    command: str


def run_make_target(
    target: str, makefile_dir: str = None, makefile_name: str = "Makefile"
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


class TestCodeToolResult(ToolResult):
    error: Error | None = None


class TestCodeTool(Tool):
    name: str = "test_code"

    def run(self, root_dir: str) -> TestCodeToolResult:
        try:
            make_result = run_make_target("test", makefile_dir=root_dir)
            if not make_result.success:
                return TestCodeToolResult(
                    error=Error(
                        description="Tests failed. Please fix the tests before proceeding.",
                        data=make_result.model_dump(),
                    )
                )
            return TestCodeToolResult(error=None)
        except Exception as e:
            return TestCodeToolResult(
                error=Error(
                    description="An unexpected error occurred while testing code.",
                    data=str(e),
                )
            )

    def get_schema(self) -> str | Callable:
        def test_code(root_dir: str) -> TestCodeToolResult:
            """
            Tests the code to ensure it is working as expected.

            This tool can be used to perform validation of any code changes made.

            Args:
                root_dir: The root directory of the project
            Returns:
                TestCodeToolResult: Result that indicates whether the code was successfully tested.
            """
            pass

        return test_code


class LintCodeToolResult(ToolResult):
    error: Error | None = None


class LintCodeTool(Tool):
    name: str = "lint_code"

    def run(self, root_dir: str) -> LintCodeToolResult:
        try:
            make_result = run_make_target("lint-check", makefile_dir=root_dir)
            if not make_result.success:
                return LintCodeToolResult(
                    error=Error(
                        description="Lint check failed. Please fix the lint errors before proceeding.",
                        data=make_result.model_dump(),
                    )
                )
            return LintCodeToolResult(error=None)
        except Exception as e:
            return LintCodeToolResult(
                error=Error(
                    description="An unexpected error occurred while running the lint check.",
                    data=str(e),
                )
            )

    def get_schema(self) -> str | Callable:
        def lint_code(root_dir: str) -> LintCodeToolResult:
            """
            Runs the lint check on the source code

            This tool can be used to perform validation of any code changes made.

            Args:
                root_dir: The root directory of the project
            Returns:
                LintCodeToolResult: Result that indicates whether the code was successfully linted.
            """
            pass

        return lint_code


class FormatCodeToolResult(ToolResult):
    error: Error | None = None


class FormatCodeTool(Tool):
    name: str = "format_code"

    def run(self, root_dir: str) -> FormatCodeToolResult:
        try:
            make_result = run_make_target("lint-fmt", makefile_dir=root_dir)
            if not make_result.success:
                return FormatCodeToolResult(
                    error=Error(
                        description="Lint formatting failed. Please manually fix the lint errors before proceeding.",
                        data=make_result.model_dump(),
                    )
                )
            return FormatCodeToolResult(error=None)
        except Exception as e:
            return FormatCodeToolResult(
                error=Error(
                    description="An unexpected error occurred while formatting code.",
                    data=str(e),
                )
            )

    def get_schema(self) -> str | Callable:
        def format_code(root_dir: str) -> FormatCodeToolResult:
            """
            Formats the source code to follow linting standards

            This tool can be used to try to automatically fix linting errors.
            Some linting errors cannot be fixed by this tool and will require
            explicitly making code changes.

            Args:
                root_dir: The root directory of the project
            Returns:
                FormatCodeToolResult: Result that indicates whether the code was successfully formatted.
            """
            pass

        return format_code
