"""
Lesson 4: Prompt Templates & Output Parsers
"""

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, CommaSeparatedListOutputParser
from pydantic import BaseModel, Field
from typing import List


# ============================================================================
# PART 1: BASIC PROMPT TEMPLATES
# ============================================================================

def example_basic_template():
    """Basic string template with single variable."""
    print("\n=== Example 1: Basic Template ===")
    
    template = PromptTemplate(
        template="Tell me a {adjective} joke about {topic}.",
        input_variables=["adjective", "topic"]
    )
    
    prompt = template.format(adjective="funny", topic="programmers")
    print(f"Generated Prompt: {prompt}")
    
    llm = OllamaLLM(model="llama3.2", temperature=0.7)
    response = llm.invoke(prompt)
    print(f"Response: {response}")


def example_chat_template():
    """Chat template with system and human messages."""
    print("\n=== Example 2: Chat Template ===")
    
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are a {role}. Be {tone} in your responses."),
        ("human", "{question}")
    ])
    
    prompt = chat_template.format_messages(
        role="Python expert",
        tone="concise and technical",
        question="What is a decorator?"
    )
    
    print(f"Generated Messages: {prompt}")


def example_partial_variables():
    """Template with partial variables (pre-filled defaults)."""
    print("\n=== Example 3: Partial Variables ===")
    
    template = PromptTemplate(
        template="{greeting}, {name}! Today is {day}.",
        input_variables=["name"],
        partial_variables={"greeting": "Hello", "day": "Monday"}
    )
    
    prompt = template.format(name="Alice")
    print(f"Generated Prompt: {prompt}")


def example_few_shot_template():
    """Few-shot learning with examples."""
    print("\n=== Example 4: Few-Shot Template ===")
    
    examples = [
        {"word": "happy", "antonym": "sad"},
        {"word": "tall", "antonym": "short"},
        {"word": "hot", "antonym": "cold"}
    ]
    
    example_template = PromptTemplate(
        template="Word: {word}\nAntonym: {antonym}",
        input_variables=["word", "antonym"]
    )
    
    few_shot_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_template,
        prefix="Give the antonym of each word:",
        suffix="Word: {input}\nAntonym:",
        input_variables=["input"]
    )
    
    prompt = few_shot_template.format(input="big")
    print(f"Generated Prompt:\n{prompt}")


# ============================================================================
# PART 2: OUTPUT PARSERS
# ============================================================================

def example_string_output_parser():
    """Simple string output parser (default)."""
    print("\n=== Example 5: String Output Parser ===")
    
    template = PromptTemplate(
        template="Give me a one-sentence answer: {question}",
        input_variables=["question"]
    )
    
    parser = StrOutputParser()
    
    llm = OllamaLLM(model="llama3.2", temperature=0.7)
    prompt = template.format(question="What is Python?")
    response = llm.invoke(prompt)
    parsed = parser.parse(response)
    
    print(f"Raw Response: {response}")
    print(f"Parsed Output (string): {parsed}")


def example_json_output_parser():
    """Parse JSON output from LLM."""
    print("\n=== Example 6: JSON Output Parser ===")
    
    parser = JsonOutputParser()
    
    template = PromptTemplate(
        template="""Generate a JSON object with information about a person.
Include: name, age, occupation, hobbies (list).
Person: {person}

{format_instructions}""",
        input_variables=["person"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    llm = OllamaLLM(model="llama3.2", temperature=0.7)
    prompt = template.format(person="a software engineer named John")
    response = llm.invoke(prompt)
    
    print(f"Raw Response: {response}")
    
    try:
        parsed = parser.parse(response)
        print(f"Parsed JSON: {json.dumps(parsed, indent=2)}")
    except Exception as e:
        print(f"Parse Error: {e}")


def example_pydantic_output_parser():
    """Parse output into Pydantic models for type safety."""
    print("\n=== Example 7: Pydantic Output Parser ===")
    
    # Define the data model
    class Person(BaseModel):
        name: str = Field(description="Person's full name")
        age: int = Field(description="Person's age in years")
        occupation: str = Field(description="Person's job title")
        hobbies: List[str] = Field(description="List of hobbies")
    
    parser = PydanticOutputParser(pydantic_object=Person)
    
    template = PromptTemplate(
        template="""Generate information about a person.
Person: {person}

{format_instructions}""",
        input_variables=["person"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    prompt = template.format(person="a 30-year-old data scientist named Sarah")
    print(f"Prompt:\n{prompt}\n")
    
    # Simulate LLM response (in practice, use llm.invoke)
    mock_response = '''```json
{
    "name": "Sarah Johnson",
    "age": 30,
    "occupation": "Data Scientist",
    "hobbies": ["machine learning", "hiking", "photography"]
}
```'''
    
    try:
        parsed = parser.parse(mock_response)
        print(f"Parsed as Pydantic Object:")
        print(f"  Name: {parsed.name}")
        print(f"  Age: {parsed.age}")
        print(f"  Occupation: {parsed.occupation}")
        print(f"  Hobbies: {', '.join(parsed.hobbies)}")
    except Exception as e:
        print(f"Parse Error: {e}")


def example_comma_separated_list_parser():
    """Parse comma-separated lists."""
    print("\n=== Example 8: Comma-Separated List Parser ===")
    
    parser = CommaSeparatedListOutputParser()
    
    template = PromptTemplate(
        template="""List 5 {topic}.
{format_instructions}""",
        input_variables=["topic"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    llm = OllamaLLM(model="llama3.2", temperature=0.7)
    prompt = template.format(topic="programming languages")
    response = llm.invoke(prompt)
    
    print(f"Raw Response: {response}")
    
    try:
        parsed = parser.parse(response)
        print(f"Parsed List: {parsed}")
        print(f"Type: {type(parsed)}")
        print(f"Items: {len(parsed)}")
    except Exception as e:
        print(f"Parse Error: {e}")


def example_datetime_output_parser():
    """Parse datetime strings using custom parser."""
    print("\n=== Example 9: Custom Datetime Parser ===")
    
    from langchain_core.output_parsers import BaseOutputParser
    from datetime import datetime
    
    class SimpleDateParser(BaseOutputParser[datetime]):
        """Parse datetime strings in ISO format."""
        
        def parse(self, text: str) -> datetime:
            """Parse ISO format datetime string."""
            # Extract date from text if needed
            text = text.strip()
            try:
                # Try ISO format
                return datetime.fromisoformat(text.replace('Z', '+00:00'))
            except:
                # Try common date formats
                for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"]:
                    try:
                        return datetime.strptime(text, fmt)
                    except:
                        continue
                raise ValueError(f"Could not parse date: {text}")
    
    parser = SimpleDateParser()
    
    # Mock response
    mock_response = "1991-02-20"
    
    try:
        parsed = parser.parse(mock_response)
        print(f"Raw Response: {mock_response}")
        print(f"Parsed Datetime: {parsed}")
        print(f"Type: {type(parsed)}")
        print(f"Year: {parsed.year}")
        print(f"Month: {parsed.month}")
        print(f"Day: {parsed.day}")
    except Exception as e:
        print(f"Parse Error: {e}")


def example_custom_output_parser():
    """Create a custom output parser."""
    print("\n=== Example 10: Custom Output Parser ===")
    
    from langchain_core.output_parsers import BaseOutputParser
    
    class BulletPointParser(BaseOutputParser[List[str]]):
        """Parse bullet points into a list."""
        
        def parse(self, text: str) -> List[str]:
            """Parse text with bullet points."""
            lines = text.strip().split('\n')
            items = []
            for line in lines:
                line = line.strip()
                # Remove bullet point markers
                if line.startswith(('- ', '* ', '• ')):
                    items.append(line[2:].strip())
                elif line.startswith(tuple(f"{i}." for i in range(1, 10))):
                    items.append(line[3:].strip())
            return items
    
    parser = BulletPointParser()
    
    mock_response = """
    - First item
    - Second item
    - Third item
    """
    
    parsed = parser.parse(mock_response)
    print(f"Raw Response: {mock_response}")
    print(f"Parsed List: {parsed}")


# ============================================================================
# PART 3: ADVANCED TEMPLATES
# ============================================================================

def example_template_composition():
    """Compose multiple templates together."""
    print("\n=== Example 11: Template Composition ===")
    
    intro_template = PromptTemplate(
        template="You are a {role}.",
        input_variables=["role"]
    )
    
    task_template = PromptTemplate(
        template="Your task is to {task}.",
        input_variables=["task"]
    )
    
    # Combine templates
    full_prompt = intro_template.format(role="Python tutor") + "\n" + \
                  task_template.format(task="explain decorators simply")
    
    print(f"Composed Prompt:\n{full_prompt}")


def example_conditional_template():
    """Template with conditional logic."""
    print("\n=== Example 12: Conditional Template ===")
    
    def create_prompt(difficulty: str, topic: str) -> str:
        if difficulty == "beginner":
            template = "Explain {topic} in simple terms for a beginner."
        elif difficulty == "advanced":
            template = "Provide an advanced explanation of {topic} with technical details."
        else:
            template = "Explain {topic} at an intermediate level."
        
        return PromptTemplate(
            template=template,
            input_variables=["topic"]
        ).format(topic=topic)
    
    prompt1 = create_prompt("beginner", "Artifical Intelligence")
    prompt2 = create_prompt("advanced", "Artifical Intelligence")
    
    for prompt in [prompt1, prompt2]:
        model = OllamaLLM(model="llama3.2", temperature=0.7)
        response = model.invoke(prompt)
        print(f"Response for prompt:\n{prompt}\n{response}\n")
        print("-" * 40)
  
    print(f"Beginner Prompt: {prompt1}")
    print(f"Advanced Prompt: {prompt2}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all examples."""
    print("=" * 70)
    print("LANGCHAIN PROMPT TEMPLATES & OUTPUT PARSERS EXAMPLES")
    print("=" * 70)
    
    # Uncomment the examples you want to run
    
    # Basic Templates
    # example_basic_template()
    # example_chat_template()
    # example_partial_variables()
    # example_few_shot_template()
    
    # Output Parsers
    # example_string_output_parser()
    # example_json_output_parser()
    # example_pydantic_output_parser()
    # example_comma_separated_list_parser()
    # example_datetime_output_parser()
    # example_custom_output_parser()
    
    # Advanced
    # example_template_composition()
    example_conditional_template()
    
    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
