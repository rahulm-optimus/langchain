# Lesson 4: Prompt Templates & Output Parsers

## 🎯 Learning Objectives

- Master prompt template creation
- Use variables effectively (input & partial)
- Implement few-shot learning
- Understand all output parser types
- Parse structured data from LLM responses
- Create custom output parsers
- Optimize prompts for better results

## 📝 What are Prompt Templates?

**Prompt Templates** are pre-defined text patterns with placeholders for dynamic content. They make prompts:
- **Reusable** - Write once, use many times
- **Consistent** - Same structure every time
- **Maintainable** - Easy to update
- **Testable** - Can validate inputs

## 🎨 Prompt Template Types

### 1. **Basic String Templates**
```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="Tell me a {adjective} joke about {topic}",
    input_variables=["adjective", "topic"]
)
```

### 2. **Chat Templates**
```python
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}"),
    ("human", "{question}")
])
```

### 3. **Few-Shot Templates**
```python
from langchain_core.prompts import FewShotPromptTemplate

examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "hot", "antonym": "cold"}
]

few_shot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="Give the antonym of each word:",
    suffix="Word: {input}\nAntonym:",
    input_variables=["input"]
)
```

## 🔧 Template Components

### Input Variables
Variables that must be provided at runtime:
```python
template = PromptTemplate(
    template="Translate {text} to {language}",
    input_variables=["text", "language"]
)
```

### Partial Variables
Pre-filled default values:
```python
template = PromptTemplate(
    template="{greeting}, {name}! Today is {day}.",
    input_variables=["name"],
    partial_variables={"greeting": "Hello", "day": "Monday"}
)
```

## 🎯 Output Parsers

Output parsers convert LLM text responses into structured data formats.

### 1. **String Output Parser**
Simple string extraction (default):
```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
result = parser.parse(llm_response)  # Returns: str
```

### 2. **JSON Output Parser**
Parse JSON objects:
```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()
template = PromptTemplate(
    template="Generate JSON: {query}\n{format_instructions}",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
result = parser.parse(llm_response)  # Returns: dict
```

### 3. **Pydantic Output Parser**
Type-safe parsing with validation:
```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="Person's name")
    age: int = Field(description="Person's age")
    hobbies: List[str] = Field(description="List of hobbies")

parser = PydanticOutputParser(pydantic_object=Person)
result = parser.parse(llm_response)  # Returns: Person object
# Access: result.name, result.age, result.hobbies
```

### 4. **Comma-Separated List Parser**
Parse lists from comma-separated strings:
```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()
result = parser.parse("Python, Java, JavaScript")  # Returns: ['Python', 'Java', 'JavaScript']
```

### 5. **Custom Datetime Parser**
Parse date and time strings with a custom parser:
```python
from langchain_core.output_parsers import BaseOutputParser
from datetime import datetime

class SimpleDateParser(BaseOutputParser[datetime]):
    def parse(self, text: str) -> datetime:
        return datetime.fromisoformat(text)

parser = SimpleDateParser()
result = parser.parse("2024-01-15")  # Returns: datetime object
```

### 6. **Custom Output Parser**
Create your own parser logic:
```python
from langchain_core.output_parsers import BaseOutputParser

class BulletPointParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split('\n')
        return [line.strip('- ').strip() for line in lines if line.strip().startswith('-')]

parser = BulletPointParser()
```

## 📊 Output Parser Comparison

| Parser | Input Format | Output Type | Use Case |
|--------|-------------|-------------|----------|
| `StrOutputParser` | Plain text | `str` | Simple text extraction |
| `JsonOutputParser` | JSON string | `dict` | Structured data |
| `PydanticOutputParser` | JSON string | Pydantic model | Type-safe, validated data |
| `CommaSeparatedListOutputParser` | CSV | `List[str]` | Lists of items |
| Custom (e.g., DateTime) | Any | Any | Custom parsing logic |

## 🔗 Combining Templates & Parsers

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

template = PromptTemplate(
    template="""Generate a person profile:
Name: {name}
{format_instructions}""",
    input_variables=["name"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt = template.format(name="Alice")
response = llm.invoke(prompt)
parsed_data = parser.parse(response)
```

## 💡 Best Practices

1. **Be Specific** - Clear, detailed instructions
2. **Use Examples** - Show desired output format (few-shot)
3. **Set Constraints** - Define boundaries and limits
4. **Format Instructions** - Use `get_format_instructions()` with parsers
5. **Error Handling** - Wrap parser calls in try-except
6. **Test Variations** - Iterate and improve templates
7. **Type Safety** - Use Pydantic for complex structured data
8. **Validation** - Add Field descriptions for better LLM guidance

## 📚 Advanced Topics

### Template Composition
Combine multiple templates:
```python
full_prompt = intro_template.format(role="expert") + "\n" + task_template.format(task="explain")
```

### Conditional Templates
Different templates based on conditions:
```python
template = beginner_template if level == "beginner" else advanced_template
```

### Streaming with Parsers
Parse streaming responses chunk by chunk for real-time output.

## 🔍 Common Pitfalls

- ❌ Not including format instructions for parsers
- ❌ Assuming LLM will always return valid JSON
- ❌ Missing error handling for parse failures
- ❌ Not validating parsed output
- ❌ Over-complicating templates

## 📖 See `example.py` for 12 working examples!

The example file includes:
- Basic, chat, and few-shot templates
- All 6 output parser types with examples
- Custom parser implementation
- Template composition
- Error handling patterns
