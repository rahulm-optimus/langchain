# Lesson 8: Tools and Toolkits

## 🎯 Learning Objectives

- Understand LangChain tools and toolkits
- Use built-in tools effectively
- Load and configure toolkits
- Combine multiple tools
- Best practices for tool usage

## 🛠️ What are Tools?

**Tools** are functions that agents can use to interact with the world. They provide specific capabilities like:
- Searching the web
- Performing calculations
- Querying databases
- Making API calls
- Reading files

### Tool Components:
1. **Name**: Identifier for the tool
2. **Description**: Tells the agent what it does
3. **Function**: The actual code to execute
4. **Input Schema**: Expected input format

## 📦 What are Toolkits?

**Toolkits** are collections of related tools. They make it easy to give agents multiple related capabilities at once.

Examples:
- **SQL Toolkit**: Query databases
- **Python REPL Toolkit**: Execute Python code
- **File Management Toolkit**: Read/write files
- **API Toolkits**: Interact with specific APIs

## 🔧 Built-in LangChain Tools

### 1. **Search Tools**
- `SerpAPIWrapper`: Google search
- `DuckDuckGoSearchRun`: Privacy-focused search
- `WikipediaQueryRun`: Wikipedia search

### 2. **Calculator Tools**
- `LLMMathChain`: Math problem solver
- `Calculator`: Basic arithmetic

### 3. **Python Tools**
- `PythonREPL`: Execute Python code
- `PythonAstREPLTool`: Safe Python execution

### 4. **File Tools**
- `ReadFileTool`: Read file contents
- `WriteFileTool`: Write to files
- `ListDirectoryTool`: List directory contents

### 5. **API Tools**
- `RequestsGetTool`: HTTP GET requests
- `RequestsPostTool`: HTTP POST requests
- `APIChain`: Query APIs with natural language

## 🎓 Creating Tools

### Method 1: Using Tool Class
```python
from langchain.tools import Tool

tool = Tool(
    name="ToolName",
    func=my_function,
    description="Clear description"
)
```

### Method 2: Using @tool Decorator
```python
from langchain.tools import tool

@tool
def my_tool(input: str) -> str:
    """Tool description here."""
    return result
```

### Method 3: Using StructuredTool
```python
from langchain.tools import StructuredTool

tool = StructuredTool.from_function(
    func=my_function,
    name="ToolName",
    description="Description"
)
```

## 📊 Tool Best Practices

### 1. **Clear Descriptions**
❌ Bad: "Does stuff"
✅ Good: "Calculates the square root of a positive number. Input: a number"

### 2. **Single Purpose**
Each tool should do one thing well

### 3. **Error Handling**
Always handle errors gracefully
```python
def my_tool(input: str) -> str:
    try:
        # Tool logic
        return result
    except Exception as e:
        return f"Error: {str(e)}"
```

### 4. **Input Validation**
Validate inputs before processing

### 5. **Descriptive Names**
Use clear, descriptive names

## 🔍 Popular Toolkits

### SQL Database Toolkit
```python
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=llm
)
tools = toolkit.get_tools()
```

### Python REPL Toolkit
```python
from langchain.agents.agent_toolkits import PythonREPLToolkit

toolkit = PythonREPLToolkit()
tools = toolkit.get_tools()
```

### JSON Toolkit
```python
from langchain.agents.agent_toolkits import JsonToolkit

toolkit = JsonToolkit()
tools = toolkit.get_tools()
```

## 💡 Tool Categories

### Information Retrieval
- Search engines
- Wikipedia
- Databases
- APIs

### Computation
- Calculators
- Python execution
- Data analysis

### File Operations
- Read files
- Write files
- Manage directories

### Communication
- Send emails
- Make API calls
- Web requests

## 🎯 Choosing the Right Tools

### Consider:
1. **Task Requirements**: What capabilities are needed?
2. **Safety**: Is the tool safe to use?
3. **Reliability**: Will it work consistently?
4. **Cost**: Are there API costs?
5. **Latency**: How fast does it need to be?

## 🔒 Tool Safety

### Dangerous Tools:
- File system access (WriteFile)
- Code execution (PythonREPL)
- System commands

### Safety Measures:
1. Sandbox execution environments
2. Input validation
3. Permission systems
4. Rate limiting
5. Audit logging

## 📝 Tool Documentation Format

```python
def my_tool(param1: str, param2: int) -> str:
    """
    Brief one-line description.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
    """
    pass
```

## 🎓 Code Examples

The `example.py` demonstrates:
- Using built-in tools
- Creating custom tools
- Working with toolkits
- Tool error handling
- Combining tools effectively

## 🐛 Common Issues

### Issue: Agent doesn't use tool
**Solution**: Improve tool description, make it more specific

### Issue: Tool fails silently
**Solution**: Add error handling and logging

### Issue: Wrong tool selected
**Solution**: Make descriptions more distinct

### Issue: Tool input format errors
**Solution**: Clearly specify input format in description

## 🎯 Success Criteria

You're ready for the next lesson when you can:
- ✅ Use built-in LangChain tools
- ✅ Create custom tools
- ✅ Work with toolkits
- ✅ Write effective tool descriptions
- ✅ Handle tool errors properly

## 🚀 Next Lesson

In Lesson 9, we'll learn how to create sophisticated custom tools tailored to your specific needs!
