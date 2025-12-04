"""
Lesson 7: Agents Basics
Creating and using AI agents
"""

import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

load_dotenv()

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    num_ctx=2048,
    num_predict=512
)

def example_1_simple_agent():
    """Example 1: Creating a simple agent with basic tools"""
    print("=" * 50)
    print("Example 1: Simple Agent with Tools")
    print("=" * 50)
   
    # Define simple tools using @tool decorator
    @tool
    def get_word_length(word: str) -> int:
        """Returns the length of a word. Input should be a single word."""
        return len(word)
    
    @tool
    def reverse_string(text: str) -> str:
        """Reverses a string. Input should be a string to reverse."""
        return text[::-1]
    
    # Create tool list
    tools = [get_word_length, reverse_string]
        
    # Create agent executor using LangGraph (modern approach)
    agent_executor = create_react_agent(llm, tools)
    
    # Test the agent
    print("\n🤖 Agent Task:\n")
    
    prompt = input("Ask to use the string tools: ")
    result = agent_executor.invoke({
        "messages": [("human", prompt)]
    })
    
    # Extract final answer from messages
    final_message = result["messages"][-1]
    print(f"\n✅ Final Answer: {final_message.content}\n")

def example_2_calculator_agent():
    """Example 2: Agent with calculator tool"""
    print("=" * 50)
    print("Example 2: Calculator Agent")
    print("=" * 50)
    
    # Simple calculator function with better description
    @tool
    def calculator(expression: str) -> str:
        """Useful for evaluating mathematical expressions. 
        Input should be a valid Python mathematical expression like:
        - '2 + 2' for addition
        - '10 * 5' for multiplication  
        - '100 / 4' for division
        - '2 ** 3' for exponentiation
        Examples: '85000 * 0.5 / 100' or '(10 + 5) * 2'
        """
        try:
            # Safely evaluate the mathematical expression
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    tools = [calculator]

    # Use create_react_agent (modern approach)
    agent_executor = create_react_agent(llm, tools)

    user_prompt = input("Enter a math question for the agent: ")

    print(f"\n🤖 Agent Task: {user_prompt}")
    
    result = agent_executor.invoke({
        "messages": [("human", user_prompt)]
    })

    # Extract final answer from messages
    final_message = result["messages"][-1]
    final_message = result.content
    print(f"\n✅ Final Answer: {final_message}\n")


def example_3_multi_tool_agent():
    """Example 3: Agent with multiple different tools"""
    print("=" * 50)
    print("Example 3: Multi-Tool Agent")
    print("=" * 50)
    
    # Define various tools
    @tool
    def get_length(text: str) -> int:
        """Returns the length of text. Input should be a string."""
        return len(text)
    
    @tool
    def to_uppercase(text: str) -> str:
        """Converts text to uppercase. Input should be a string."""
        return text.upper()
    
    @tool
    def count_words(text: str) -> int:
        """Counts the number of words in text. Input should be a string."""
        return len(text.split())
    
    @tool
    def multiply_numbers(input_str: str) -> str:
        """Multiplies two numbers. Input format: 'num1,num2' where num1 and num2 are numbers."""
        try:
            nums = input_str.split(',')
            result = float(nums[0].strip()) * float(nums[1].strip())
            return str(result)
        except Exception as e:
            return f"Error: Please provide two numbers separated by comma. Error: {str(e)}"

    tools = [get_length, to_uppercase, count_words, multiply_numbers]

    # Create agent using LangGraph
    agent_executor = create_react_agent(llm, tools)
       
    # Complex task requiring multiple tools
    print("\n🤖 Agent Task: Complex multi-step problem")
    result = agent_executor.invoke({
        "messages": [("human", "Take the text 'Hello World', count how many words it has, "
                     "then multiply that number by 5. What's the result?")]
    })
    
    # Extract final answer from messages
    final_message = result["messages"][-1]
    print(f"\n✅ Final Answer: {final_message.content}\n")


def example_4_agent_with_memory():
    """Example 4: Conversational agent with memory"""
    print("=" * 50)
    print("Example 4: Agent with Memory")
    print("=" * 50)
       
    @tool
    def get_current_year() -> str:
        """Returns the current year (2025)."""
        return "2025"
    
    @tool
    def calculate(expression: str) -> str:
        """Calculates a mathematical expression. Input should be a valid Python expression."""
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error in calculation: {str(e)}"
    
    tools = [get_current_year, calculate]
    
    # Create agent with LangGraph (memory is built-in via message history)
    agent_executor = create_react_agent(llm, tools)
    
    # First interaction
    print("\n🤖 First question:")
    result1 = agent_executor.invoke({
        "messages": [("human", "What year is it?")]
    })
    answer1 = result1["messages"][-1].content
    print(f"Answer: {answer1}")
    
    # Second interaction - agent maintains context via message history
    print("\n🤖 Follow-up question (testing memory):")
    # Include previous messages for context
    result2 = agent_executor.invoke({
        "messages": [
            ("human", "What year is it?"),
            ("assistant", answer1),
            ("human", "What will the year be in 5 years?")
        ]
    })
    answer2 = result2["messages"][-1].content
    print(f"Answer: {answer2}\n")


def example_5_agent_error_handling():
    """Example 5: Handling agent errors gracefully"""
    print("=" * 50)
    print("Example 5: Agent Error Handling")
    print("=" * 50)
    
    @tool
    def risky_operation(input_str: str) -> str:
        """A tool that processes input but may fail on certain inputs. 
        Input should be a string to process."""
        if "error" in input_str.lower():
            raise ValueError("Intentional error for demonstration")
        return f"Success: Processed '{input_str}'"
    
    tools = [risky_operation]
    
    # Create agent with LangGraph
    agent_executor = create_react_agent(llm, tools)
    
    print("\n🤖 Test 1: Normal operation")
    try:
        result = agent_executor.invoke({
            "messages": [("human", "Use the risky_operation tool with the text 'hello'")]
        })
        final_message = result["messages"][-1]
        print(f"✅ Result: {final_message.content}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print("\n🤖 Test 2: Triggering error")
    try:
        result = agent_executor.invoke({
            "messages": [("human", "Use the risky_operation tool with the text 'error'")]
        })
        final_message = result["messages"][-1]
        print(f"✅ Result: {final_message.content}")
    except Exception as e:
        print(f"❌ Error handled: {str(e)[:100]}")
    
    print()


def main():
    """Run all examples"""
    print("\n🚀 LangChain Agents Basics Examples\n")
       
    try:
        # example_1_simple_agent()
        # example_2_calculator_agent()
        # example_3_multi_tool_agent()
        example_4_agent_with_memory()
        # example_5_agent_error_handling()
        
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
