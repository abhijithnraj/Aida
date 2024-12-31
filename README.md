# AIDA - AI Server Management Assistant

AIDA is a natural language assistant that helps you manage your server by interpreting your questions and executing appropriate commands.

## Features

- Natural language processing of server management queries
- Safe execution of shell commands
- Interactive CLI interface
- Built with LangChain and Ollama

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd aida
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure you have Ollama installed, running and the models downloaded.

```

## Usage

You can start AIDA using the CLI interface:

```bash
python -m aida.cli
```

Or use it in your Python code:

```python
from aida import Aida

aida = Aida()
response = aida.process_query("What is the current directory?")
print(response)
```



