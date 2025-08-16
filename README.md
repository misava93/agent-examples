# Agent Examples

A collection of AI agent examples demonstrating automated bug fixing and code maintenance capabilities using local LLMs via Ollama.

## Overview

This repository showcases practical examples of AI agents that can autonomously identify, diagnose, and fix bugs in software applications. The main example provided is a ticket-based bug fixing agent that processes bug reports and applies fixes automatically.

## Features

- 🐛 **Automated Bug Fixing**: AI agent that reads bug tickets and implements fixes
- 🔍 **Code Analysis**: Automatically identifies relevant files and root causes
- ✅ **Test Generation**: Creates tests to reproduce bugs before fixing them
- 🔧 **Validation**: Runs tests and linting to ensure fixes are correct
- 🤖 **Local LLM Support**: Uses Ollama for running models locally

## Project Structure

```
agent-examples/
├── agents/
│   ├── __init__.py
│   └── ticket-bug-fixer.py    # Main bug fixing agent implementation
├── main.py                     # Sample FastAPI application with intentional bug
├── requirements.txt            # Python dependencies
├── Makefile                    # Build and development commands
└── prompt.txt                  # Instructions for agent tasks
```

## Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.ai/) installed and running
- `make` command available

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/agent-examples.git
cd agent-examples
```

2. Create and activate a virtual environment:
```bash
make venv-create
source .venv/bin/activate
source .venv/bin/activate.fish  # For fish shell
```

3. Install dependencies:
```bash
make install-deps
```

4. Ensure Ollama is running with the required model:
```bash
ollama pull gpt-oss:20b
```

## Usage

### Running the Bug Fixer Agent

The ticket bug fixer agent demonstrates how AI can automatically fix bugs based on ticket descriptions:

```bash
make run-ticket-bug-fixer
```

When prompted, provide a ticket ID (any string). The agent will:
1. Retrieve the bug report
2. Analyze the codebase
3. Identify the root cause
4. Write tests to reproduce the bug
5. Implement a fix
6. Validate the fix with tests and linting

> You will see the agent's thought process, the updated context and the tool calls made at each iteration.

### Observations
- It's fun to watch the model sometimes get stuck doing the same thing over and over again, and then randomly decide to try something else.
- This implementation contains the bare minimum to have a full proof of concept able to successfully fix the bug present in this repository.
- Based on my testing, I found the following models to work best:
    - OpenAI new open weights model: `gpt-oss:20b`
    - Qwen 3 8B: `qwen3:8b`

### Sample API Application

The repository includes a sample FastAPI application with an intentional bug for testing:

```bash
# Start the API in development mode
make api-start-dev

# Test the health endpoint (will trigger the bug)
make test-api
```

## Development Commands

The Makefile provides several useful commands:

- `make venv-create` - Create a virtual environment
- `make install-deps` - Install Python dependencies
- `make lint-check` - Check code formatting with Black
- `make lint-fmt` - Format code with Black
- `make test` - Run tests with pytest
- `make api-start-dev` - Start API in development mode
- `make api-start-prod` - Start API in production mode
- `make test-api` - Test the API health endpoint
- `make run-ticket-bug-fixer` - Run the bug fixing agent

## How It Works

### Bug Fixing Agent

The bug fixing agent (`agents/ticket-bug-fixer.py`) follows this workflow:

1. **Ticket Processing**: Reads bug reports and extracts relevant information
2. **Code Discovery**: Finds relevant files in the repository
3. **Analysis**: Reads and understands the code structure
4. **Test Creation**: Writes tests to reproduce the reported bug
5. **Fix Implementation**: Modifies code to resolve the issue
6. **Validation**: Runs tests and linting to ensure quality

### Key Components

- **Context Management**: Maintains state across iterations
- **Tool Integration**: Uses various tools (file reading, editing, validation)
- **LLM Integration**: Communicates with Ollama for decision-making

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) for the sample application
- Uses [Ollama](https://ollama.ai/) for local LLM execution
- Formatted with [Black](https://github.com/psf/black) for consistent code style
- Tested with [pytest](https://pytest.org/) for reliable validation
