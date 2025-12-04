"""
Lesson 2: Setup and Basics
Basic LangChain setup with Ollama
"""

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_ollama import OllamaLLM

load_dotenv()

def example_1_basic_llm_call():
    """Example 1: Making a basic call to an LLM with Ollama"""
    print("=" * 50)
    print("Example 1: Basic LLM Call")
    print("=" * 50)
    
    try:
        # Create an LLM instance (runs locally, no API key needed)
        llm = OllamaLLM(
            model="llama3.2",  # Free model
            temperature=0.7  # Controls randomness (0-1)
        )
        
        # Make a simple request
        response = llm.invoke("What is LangChain in one sentence?")
        print(f"Response: {response}\n")
        
    except ImportError:
        print("❌ langchain-ollama not installed")
        print("💡 Install: pip install langchain-ollama")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Install Ollama first: https://ollama.ai/download")
        print("   Then run: ollama pull llama3.2\n")


def example_2_chat_with_system_message():
    """Example 2: Using system and user messages"""
    print("=" * 50)
    print("Example 2: Chat with System Message")
    print("=" * 50)
    
    try:
        llm = OllamaLLM(model="llama3.2", temperature=0.7)
        
        # Ollama works with formatted text prompts
        prompt = """System: You are a helpful Python programming assistant.

User: What is an array in python?"""
        
        response = llm.invoke(prompt)
        print(f"Response: {response}\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("� Make sure Ollama is running\n")


def example_3_prompt_template():
    """Example 3: Using prompt templates for reusability"""
    print("=" * 50)
    print("Example 3: Prompt Templates")
    print("=" * 50)
    
    try:
        llm = OllamaLLM(model="llama3.2", temperature=0.7)
        
        # Create a reusable template
        template = """You are a {role} expert.
    
Question: {question}
    
Please provide a clear and concise answer."""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["role", "question"]
        )
        
        # Format the prompt with specific values
        formatted_prompt = prompt.format(
            role="Python programming",
            question="What are decorators?"
        )
        
        print(f"Formatted Prompt:\n{formatted_prompt}\n")
        
        response = llm.invoke(formatted_prompt)
        print(f"Response: {response}\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure Ollama is installed and llama3.2 is pulled\n")


def example_4_chat_prompt_template():
    """Example 4: Using ChatPromptTemplate for structured conversations"""
    print("=" * 50)
    print("Example 4: Chat Prompt Template")
    print("=" * 50)
    
    try:
        llm = OllamaLLM(model="llama3.2", temperature=0.7)
        
        # Create a chat template
        chat_template = ChatPromptTemplate.from_messages([
            ("system", "You are a {expertise} tutor. Explain concepts simply."),
            ("human", "Explain {topic} with an example.")
        ])
        
        # Format and invoke
        messages = chat_template.format_messages(
            expertise="machine learning",
            topic="supervised learning"
        )
        
        # Convert messages to string format for Ollama
        prompt_text = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
        
        response = llm.invoke(prompt_text)
        print(f"Response: {response}\n")
        
    except Exception as e:
        print(f"❌ Error: {e}\n")


def main():   
    try:
        """Main function to present options and execute selected functions."""

        while True:
            print("\n--- LLM Example Options ---")
            print("1. Example 1: Basic LLM Call")
            print("2. Example 2: Chat with System Message")
            print("3. Example 3: Prompt Template")
            print("4. Example 4: Chat Prompt Template")
            print("5. Exit")

            choice = input("Enter your choice (1-5): ")

            if choice == '1':
                example_1_basic_llm_call()
                continue  # Go back to the beginning to ask for another option
            elif choice == '2':
                example_2_chat_with_system_message()
                continue
            elif choice == '3':
                example_3_prompt_template()
                continue
            elif choice == '4':
                example_4_chat_prompt_template()
                continue
            elif choice == '5':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 5.")
                       
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure Ollama is installed: https://ollama.ai/download")
        print("   2. Pull the model: ollama pull llama3.2")
        print("   3. Check if Ollama is running")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
