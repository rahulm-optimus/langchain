# Lesson 9: Custom Tools

## 🎯 Learning Objectives

- Create sophisticated custom tools
- Handle complex inputs and outputs
- Implement error handling and validation
- Build tools that interact with external systems
- Design tool interfaces for agents

## 🛠️ Why Custom Tools?

Built-in tools are great, but real applications need specialized capabilities:
- Domain-specific operations
- Integration with your systems
- Proprietary data access
- Custom business logic
- Unique workflows

## 🎨 Custom Tool Design Principles

### 1. **Single Responsibility**
Each tool should do ONE thing well
```python
# ❌ Bad: Too many responsibilities
def do_everything_tool(input): ...

# ✅ Good: Focused tools
def calculate_tax(amount): ...
def validate_email(email): ...
```

### 2. **Clear Description**
The agent relies on descriptions to choose tools
```python
# ❌ Bad
"Does stuff with numbers"

# ✅ Good
"Calculates the compound interest given principal, rate, and time."
```

### 3. **Robust Error Handling**
Always catch and handle errors
```python
@tool
def my_tool(input: str) -> str:
    """Tool description."""
    try:
        # Tool logic
        return result
    except ValueError as e:
        return f"Invalid input: {str(e)}"
    except Exception as e:
        return f"Error occurred: {str(e)}"
```

### 4. **Input Validation**
Validate before processing
```python
def validate_email_tool(email: str) -> str:
    """Validates an email address."""
    if '@' not in email:
        return "Invalid: Email must contain @"
    # Proceed with validation
```

## 🏗️ Building Complex Custom Tools

### Example: Database Query Tool
```python
@tool
def query_customer_database(customer_id: str) -> str:
    """
    Queries the customer database by ID.
    Input: customer ID (e.g., 'CUST123')
    Returns: customer information or error
    """
    try:
        # Connect to database
        # Execute query
        # Return formatted result
        return result
    except Exception as e:
        return f"Database error: {str(e)}"
```

### Example: File Processing Tool
```python
@tool
def process_csv_file(filepath: str) -> str:
    """
    Processes a CSV file and returns summary statistics.
    Input: path to CSV file
    Returns: row count, column names, and basic stats
    """
    try:
        import pandas as pd
        df = pd.read_csv(filepath)
        return f"Rows: {len(df)}, Columns: {list(df.columns)}"
    except Exception as e:
        return f"Error reading file: {str(e)}"
```

## 🔧 Advanced Tool Patterns

### 1. **Stateful Tools**
Tools that maintain state between calls
```python
class CounterTool:
    def __init__(self):
        self.count = 0
    
    def increment(self) -> str:
        """Increments the counter."""
        self.count += 1
        return f"Count is now {self.count}"
```

### 2. **Async Tools**
For I/O-bound operations
```python
@tool
async def fetch_data_async(url: str) -> str:
    """Fetches data from a URL asynchronously."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
```

### 3. **Tools with Multiple Parameters**
Use StructuredTool for complex inputs
```python
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(description="Maximum results")

tool = StructuredTool.from_function(
    func=search_function,
    name="Search",
    description="Searches with query and limit",
    args_schema=SearchInput
)
```

## 💾 Tools for Data Access

### Database Tools
```python
@tool
def execute_sql_query(query: str) -> str:
    """
    Executes a SQL query (READ ONLY).
    Input: SELECT query
    Returns: query results
    """
    # Validate query is SELECT only
    # Execute safely
    # Return formatted results
```

### API Integration Tools
```python
@tool
def call_internal_api(endpoint: str) -> str:
    """
    Calls internal company API.
    Input: endpoint path
    Returns: API response
    """
    base_url = "https://api.company.com"
    response = requests.get(f"{base_url}/{endpoint}")
    return response.json()
```

## 🧪 Testing Custom Tools

### Unit Testing
```python
def test_calculator_tool():
    result = calculator_tool("2+2")
    assert "4" in result
    
def test_error_handling():
    result = calculator_tool("invalid")
    assert "Error" in result
```

### Integration Testing
```python
def test_tool_with_agent():
    agent = create_agent(tools=[my_custom_tool])
    result = agent.invoke("Use my tool")
    assert result is not None
```

## 🔒 Security Considerations

### 1. **Input Sanitization**
```python
def sanitize_input(user_input: str) -> str:
    # Remove dangerous characters
    # Validate format
    # Escape special characters
    return clean_input
```

### 2. **Permission Checks**
```python
@tool
def access_sensitive_data(user_id: str) -> str:
    """Accesses sensitive data with permission check."""
    if not has_permission(user_id):
        return "Access denied"
    return get_data(user_id)
```

### 3. **Rate Limiting**
```python
from functools import lru_cache
import time

@tool
@rate_limit(calls=10, period=60)
def rate_limited_tool(input: str) -> str:
    """Tool with rate limiting."""
    pass
```

## 📊 Tool Performance

### Monitoring
- Track execution time
- Log errors and successes
- Monitor usage patterns

### Optimization
- Cache frequently used results
- Optimize database queries
- Use async for I/O operations

## 🎓 Code Examples

The `example.py` demonstrates:
- Creating custom tools from scratch
- Tools with validation
- Database interaction tools
- API integration tools
- Error handling patterns
- Complex multi-parameter tools

## 🐛 Common Issues

### Issue: Tool not being used
**Solution**: Improve description, make it more specific about when to use

### Issue: Input format errors
**Solution**: Clearly document expected input format in description

### Issue: Tool timeout
**Solution**: Implement timeout handling, use async for long operations

### Issue: State management
**Solution**: Use class-based tools or external state storage

## 💡 Best Practices Checklist

- ✅ Clear, descriptive tool name
- ✅ Comprehensive description
- ✅ Input validation
- ✅ Error handling
- ✅ Type hints
- ✅ Documentation
- ✅ Unit tests
- ✅ Logging
- ✅ Security checks
- ✅ Performance optimization

## 🎯 Success Criteria

You're ready for the next lesson when you can:
- ✅ Design effective custom tools
- ✅ Implement robust error handling
- ✅ Validate inputs properly
- ✅ Integrate with external systems
- ✅ Test tools thoroughly
- ✅ Handle security concerns

## 🚀 Next Lesson

In Lesson 10, we'll explore Advanced Agents including specialized agent types, agent chains, and complex agent architectures!
