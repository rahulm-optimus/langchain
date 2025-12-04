"""
Lesson 8: Tools and Toolkits
"""

import os
from dotenv import load_dotenv
import requests
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

load_dotenv()

model = os.getenv("model_2")

llm = ChatOllama(
    model=model,
    temperature=0,
    num_ctx=2048,
    num_predict=512
)


def example_1_basic_tools():
    """Example 1: Creating and using basic tools"""
    print("=" * 50)
    print("Example 1: Basic Tools")
    print("=" * 50)
      
   # Better tool definitions with clear descriptions
    @tool
    def add_numbers(input_str: str) -> str:
        """Add two numbers together.
        
        This tool adds two numbers and returns the sum.
        
        Args:
            input_str: Two numbers separated by a comma (e.g., "34,45" or "5.5,3.2")
            
        Returns:
            A string with the sum of the two numbers
            
        Examples:
            - Input: "34,45" -> Output: "The sum is 79.0"
            - Input: "10,20" -> Output: "The sum is 30.0"
        """
        try:
            # Clean and split the input
            nums = [n.strip() for n in input_str.split(',')]
            if len(nums) != 2:
                return "Error: Please provide exactly two numbers separated by a comma (e.g., '34,45')"
            
            result = float(nums[0]) + float(nums[1])
            return f"The sum is {result}"
        except ValueError:
            return "Error: Please provide valid numbers (e.g., '34,45')"
        except Exception as e:
            return f"Error: {str(e)}"
    
    @tool
    def multiply_numbers(input_str: str) -> str:
        """Multiply two numbers together.
        
        This tool multiplies two numbers and returns the product.
        
        Args:
            input_str: Two numbers separated by a comma (e.g., "34,45" or "5.5,3.2")
            
        Returns:
            A string with the product of the two numbers
            
        Examples:
            - Input: "34,45" -> Output: "The product is 1530.0"
            - Input: "10,5" -> Output: "The product is 50.0"
        """
        try:
            # Clean and split the input
            nums = [n.strip() for n in input_str.split(',')]
            if len(nums) != 2:
                return "Error: Please provide exactly two numbers separated by a comma (e.g., '34,45')"
            
            result = float(nums[0]) * float(nums[1])
            return f"The product is {result}"
        except ValueError:
            return "Error: Please provide valid numbers (e.g., '34,45')"
        except Exception as e:
            return f"Error: {str(e)}"
    
    tools = [add_numbers, multiply_numbers]
    
    
    # Create agent using LangGraph (no prompt needed)
    agent_executor = create_react_agent(llm, tools)

    prompt = input("Enter a math task ")
    print(f"\n🤖 Task: {prompt}")
    result = agent_executor.invoke({
        "messages": [("human", prompt)]
    })
    
    # Extract final answer
    final_message = result["messages"][-1]
    print(f"\n✅ Answer: {final_message.content}\n")

def example_2_text_processing_tools():
    """Example 2: Tools for text processing"""
    print("=" * 50)
    print("Example 2: Text Processing Tools")
    print("=" * 50)
    
    @tool
    def count_characters(text: str) -> str:
        """Counts the number of characters in text."""
        return f"Character count: {len(text)}"
    
    @tool
    def count_words(text: str) -> str:
        """Counts the number of words in text."""
        return f"Word count: {len(text.split())}"
    
    @tool
    def reverse_text(text: str) -> str:
        """Reverses the given text."""
        return f"Reversed: {text[::-1]}"
    
    @tool
    def to_uppercase(text: str) -> str:
        """Converts text to uppercase."""
        return text.upper()
    
    @tool
    def find_vowels(text: str) -> str:
        """Counts vowels in the text."""
        vowels = 'aeiouAEIOU'
        count = sum(1 for char in text if char in vowels)
        return f"Vowel count: {count}"
    
    tools = [
        count_characters,
        count_words,
        reverse_text,
        to_uppercase,
        find_vowels
    ]
    
    # Create agent using LangGraph
    agent_executor = create_react_agent(llm, tools)
    
    print("\n🤖 Task: Analyze the text 'LangChain'")
    result = agent_executor.invoke({
        "messages": [("human", "For the text 'LangChain', tell me: how many characters, how many words, and how many vowels it has.")]
    })
    
    # Extract final answer
    final_message = result["messages"][-1]
    print(f"\n✅ Answer: {final_message.content}\n")


def example_3_data_tools():
    """Example 3: Tools for data manipulation"""
    print("=" * 50)
    print("Example 3: Data Manipulation Tools")
    print("=" * 50)
    
    @tool
    def calculate_average(numbers_str: str) -> str:
        """
        Calculates the average of numbers.
        Input format: comma-separated numbers (e.g., '1,2,3,4,5')
        """
        try:
            numbers = [float(n.strip()) for n in numbers_str.split(',')]
            avg = sum(numbers) / len(numbers)
            return f"Average: {avg:.2f}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    @tool
    def find_max(numbers_str: str) -> str:
        """
        Finds the maximum number.
        Input format: comma-separated numbers (e.g., '1,2,3,4,5')
        """
        try:
            numbers = [float(n.strip()) for n in numbers_str.split(',')]
            return f"Maximum: {max(numbers)}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    @tool
    def find_min(numbers_str: str) -> str:
        """
        Finds the minimum number.
        Input format: comma-separated numbers (e.g., '1,2,3,4,5')
        """
        try:
            numbers = [float(n.strip()) for n in numbers_str.split(',')]
            return f"Minimum: {min(numbers)}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    @tool
    def sort_numbers(numbers_str: str) -> str:
        """
        Sorts numbers in ascending order.
        Input format: comma-separated numbers (e.g., '5,2,8,1,9')
        """
        try:
            numbers = [float(n.strip()) for n in numbers_str.split(',')]
            sorted_nums = sorted(numbers)
            return f"Sorted: {', '.join(map(str, sorted_nums))}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    tools = [calculate_average, find_max, find_min, sort_numbers]
    
    # Fixed: Create agent properly
    agent = create_react_agent(llm, tools)
       
    print("\n🤖 Task: Analyze numbers 15, 23, 8, 42, 16")
    result = agent.invoke({
        "messages": [("human", "For the numbers 15, 23, 8, 42, 16, find the average, maximum, and sort them.")]
    })
    
    # Fixed: Extract final message correctly
    final_message = result['messages'][-1]
    print(f"\n✅ Answer: {final_message.content}\n")

    # Fixed: Follow-up question (if needed)
    if final_message.content:
        print("\n🤔 Getting interpretation...")
        
        # Just use LLM directly for follow-up (no agent needed)
        follow_up_prompt = f"""Based on this data analysis result:{final_message.content}
        Please explain what these statistics tell us about the dataset."""
        
        response = llm.invoke(follow_up_prompt)
        print(f"\n💡 Interpretation: {response.content}\n")


def example_4_api_tools():
    """Example 4: Tools that interact with APIs"""
    print("=" * 50)
    print("Example 4: API Tools")
    print("=" * 50)
    
    # -----------------------------------------------------
    # 1. Helper function to fetch and cache user data
    # -----------------------------------------------------
    def fetch_all_users():
        """Fetch all users once per request."""
        resp = requests.get("https://dummyjson.com/users")
        resp.raise_for_status()
        return resp.json()["users"]


    # -----------------------------------------------------
    # 2. Tools
    # -----------------------------------------------------

    @tool
    def filter_users_by_age(criteria: str) -> dict:
        """
        Filter users by age.
        Examples:
            - 'less than 30'
            - '< 25'
            - 'greater than 40'
            - '= 18'
        Returns clean JSON output.
        """
        criteria = criteria.lower().strip()

        # detect operator
        if "less" in criteria or "<" in criteria:
            op = "less"
        elif "greater" in criteria or ">" in criteria:
            op = "greater"
        elif "equal" in criteria or "=" in criteria or criteria.isdigit():
            op = "equal"
        else:
            return {"error": "Invalid age format. Use 'less than X', 'greater than X', or '= X'."}

        # extract number
        digits = "".join([c for c in criteria if c.isdigit()])
        if digits == "":
            return {"error": "Age number not found."}
        value = int(digits)

        users = fetch_all_users()

        # filtering
        if op == "less":
            filtered = [u for u in users if u["age"] < value]
        elif op == "greater":
            filtered = [u for u in users if u["age"] > value]
        else:
            filtered = [u for u in users if u["age"] == value]

        return {
            "criteria": criteria,
            "count": len(filtered),
            "users": filtered
        }


    @tool
    def get_user_by_id(user_id: int) -> dict:
        """Return user matching a specific ID."""
        try:
            user_id = int(user_id)
            users = fetch_all_users()
            for u in users:
                if u["id"] == user_id:
                    return u
            return {"error": f"No user found with id {user_id}"}
        except:
            return {"error": "User ID must be a number."}


    @tool
    def search_users_by_city(city: str) -> dict:
        """Return users who live in the given city."""
        city = city.lower()
        users = fetch_all_users()
        filtered = [u for u in users if u["address"]["city"].lower() == city]

        return {
            "city": city,
            "count": len(filtered),
            "users": filtered
        }


    @tool
    def count_users_by_gender(gender: str) -> str:
        """Counts and lists ALL users by gender. Ex: 'male' or 'female'."""
        gender_lower = gender.lower()
        users = fetch_all_users()
        filtered = [u for u in users if u["gender"].lower() == gender_lower]

        if not filtered:
            return f"No {gender} users found"

        result = [f"Found {len(filtered)} {gender} users:\n"]
        for i, u in enumerate(filtered, 1):
            result.append(
                f"{i}. {u['firstName']} {u['lastName']} "
                f"(Age: {u['age']}, City: {u['address']['city']})"
            )
        
        return "\n".join(result)


    # -----------------------------------------------------
    # 3. Create agent
    # -----------------------------------------------------
    tools = [filter_users_by_age, get_user_by_id, search_users_by_city, count_users_by_gender]

    agent = create_react_agent(llm, tools)

    query = input("\nEnter your question about users: ")

    result = agent.invoke({
        "messages": [("human", query)]
    })

    final_msg = result["messages"][-1].content

    print("\n\n✅ Final Result:")
    print(final_msg)

def example_5_conditional_tools():
    """Example 5: Tools with conditional logic"""
    print("=" * 50)
    print("Example 5: Conditional Logic Tools")
    print("=" * 50)
    
    @tool
    def check_even_odd(number_str: str) -> str:
        """Checks if a number is even or odd. 
        
        Args:
            number_str: A valid integer number (e.g., "17", "42", "-5")
            
        Returns:
            Whether the number is even or odd, or an error message if input is invalid
        """
        try:
            number = int(number_str)
            if number % 2 == 0:
                return f"{number} is EVEN"
            else:
                return f"{number} is ODD"
        except ValueError:
            return "ERROR: Input must be a valid integer number. Please provide a number like 17, 42, etc."
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    @tool
    def check_positive_negative(number_str: str) -> str:
        """Checks if a number is positive, negative, or zero.
        
        Args:
            number_str: A valid number (e.g., "17", "-5", "3.14")
            
        Returns:
            Whether the number is positive, negative, or zero, or an error message
        """
        try:
            number = float(number_str)
            if number > 0:
                return f"{number} is POSITIVE"
            elif number < 0:
                return f"{number} is NEGATIVE"
            else:
                return f"{number} is ZERO"
        except ValueError:
            return "ERROR: Input must be a valid number. Please provide a number like 17, -5, or 3.14."
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    @tool
    def check_prime(number_str: str) -> str:
        """Checks if a number is prime.
        
        Args:
            number_str: A positive integer (e.g., "17", "42")
            
        Returns:
            Whether the number is prime or not, or an error message
        """
        try:
            number = int(number_str)
            if number < 2:
                return f"{number} is NOT prime (prime numbers must be >= 2)"
            for i in range(2, int(number ** 0.5) + 1):
                if number % i == 0:
                    return f"{number} is NOT prime"
            return f"{number} is PRIME"
        except ValueError:
            return "ERROR: Input must be a valid positive integer. Please provide a number like 17 or 42."
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    tools = [check_even_odd, check_positive_negative, check_prime]
    
    agent = create_react_agent(llm, tools)

    print("\n🤖 Task: Check number properties")
    print("💡 Examples: 17, 42, -5, 100\n")
    
    query = input("Enter a number to analyze (or anything else): ")
    
    # Simplified prompt - let the agent figure it out
    result = agent.invoke({
        "messages": [
            ("system", """You are a number analysis assistant with access to three tools:
1. check_even_odd - determines if a number is even or odd
2. check_positive_negative - determines if a number is positive, negative, or zero
3. check_prime - determines if a number is prime

When given a NUMBER:
- Use ALL three tools to analyze it
- Report all findings clearly

When given something that is NOT a number (like text, words, empty input):
- Do NOT try to use the tools (they will return errors)
- Politely explain you can only analyze numbers
- Give examples of valid inputs (like 17, 42, -5, 100)
- Ask the user to provide a valid number"""),
            ("human", f"Analyze: {query}")
        ]
    })

    final_msg = result["messages"][-1].content

    print("\n✅ Final Result:")
    print(final_msg)

def main():
    """Run all examples"""
    print("\n🚀 LangChain Tools and Toolkits Examples\n")
    
    if not os.getenv("model_2"):
        print("❌ Error: model not found!")
        return
    
    try:
        # example_1_basic_tools()    
        # example_2_text_processing_tools()
        # example_3_data_tools()
        # example_4_api_tools()
        example_5_conditional_tools()
        
       
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
