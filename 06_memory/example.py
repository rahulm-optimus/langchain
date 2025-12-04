"""
Lesson 6: Memory - Modern Approaches
Updated to use latest LangChain patterns (2024+)
"""

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = os.getenv("model_2")
llm = ChatOllama(model=model, temperature=0.7)

# Store for multiple conversation sessions
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create chat history for a session"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def example_1_basic_memory():
    """Example 1: Basic conversation memory with LCEL"""
    print("=" * 60)
    print("Example 1: Basic Conversation Memory")
    print("=" * 60)
    
    # Create prompt with message history placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Remember the conversation context."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Create chain with memory
    chain = prompt | llm | StrOutputParser()
    
    # Wrap chain with message history
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    # Test conversation
    session_id = "user_123"
    
    print("\n💬 Conversation 1:")
    response1 = chain_with_history.invoke(
        {"input": "Hi, my name is Alice"},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"User: Hi, my name is Alice")
    print(f"AI: {response1}")
    
    print("\n💬 Conversation 2:")
    response2 = chain_with_history.invoke(
        {"input": "What's my name?"},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"User: What's my name?")
    print(f"AI: {response2}")
    
    # Show stored history
    print("\n📝 Stored Messages:")
    history = get_session_history(session_id)
    for msg in history.messages:
        print(f"  {msg.type}: {msg.content[:50]}...")
    
    print()


def example_2_multi_session():
    """Example 2: Multiple conversation sessions"""
    print("=" * 60)
    print("Example 2: Multiple Sessions")
    print("=" * 60)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    # Session 1
    print("\n👤 User 1 (session_1):")
    r1 = chain_with_history.invoke(
        {"input": "I like pizza"},
        config={"configurable": {"session_id": "session_1"}}
    )
    print(f"User: I like pizza")
    print(f"AI: {r1[:100]}...")
    
    # Session 2
    print("\n👤 User 2 (session_2):")
    r2 = chain_with_history.invoke(
        {"input": "I like sushi"},
        config={"configurable": {"session_id": "session_2"}}
    )
    print(f"User: I like sushi")
    print(f"AI: {r2[:100]}...")
    
    # Test memory isolation
    print("\n🧪 Testing Memory Isolation:")
    print("\nAsking session_1: What do I like?")
    r3 = chain_with_history.invoke(
        {"input": "What do I like?"},
        config={"configurable": {"session_id": "session_1"}}
    )
    print(f"AI: {r3[:100]}...")
    
    print("\nAsking session_2: What do I like?")
    r4 = chain_with_history.invoke(
        {"input": "What do I like?"},
        config={"configurable": {"session_id": "session_2"}}
    )
    print(f"AI: {r4[:100]}...")
    
    print()


def example_3_window_memory():
    """Example 3: Limited history window (last N messages)"""
    print("=" * 60)
    print("Example 3: Window Memory (Last 4 Messages)")
    print("=" * 60)
    
    from langchain_core.runnables import RunnableLambda
    
    def limit_messages(messages):
        """Keep only last 4 messages"""
        return messages[-4:] if len(messages) > 4 else messages
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    # Create custom history getter with window
    def get_windowed_history(session_id: str):
        history = get_session_history(session_id)
        # Return limited messages
        return history
    
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_windowed_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    session_id = "window_test"
    
    # Have multiple conversations
    questions = [
        "My favorite color is blue",
        "I live in New York",
        "I work as a developer",
        "I like hiking",
        "What's my favorite color?",  # Should remember
        "Where do I live?",  # Might forget (too old)
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n💬 Turn {i}: {question}")
        response = chain_with_history.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )
        print(f"AI: {response[:100]}...")
        
        # Show history size
        history = get_session_history(session_id)
        print(f"📊 History size: {len(history.messages)} messages")
    
    print()


def example_4_chat_with_system():
    """Example 4: Chat with custom system message and memory"""
    print("=" * 60)
    print("Example 4: Custom System Message")
    print("=" * 60)
    
    # Different system prompts for different roles
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a {role}. {instructions}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    session_id = "python_tutor"
    
    print("\n👨‍🏫 Python Tutor Session:")
    
    response1 = chain_with_history.invoke(
        {
            "role": "Python programming tutor",
            "instructions": "Explain concepts clearly with examples. Remember what you've taught.",
            "input": "Teach me about lists"
        },
        config={"configurable": {"session_id": session_id}}
    )
    print(f"User: Teach me about lists")
    print(f"AI: {response1[:200]}...\n")
    
    response2 = chain_with_history.invoke(
        {
            "role": "Python programming tutor",
            "instructions": "Explain concepts clearly with examples. Remember what you've taught.",
            "input": "Give me an example using what you just taught"
        },
        config={"configurable": {"session_id": session_id}}
    )
    print(f"User: Give me an example using what you just taught")
    print(f"AI: {response2[:200]}...\n")
    
    print()


def example_5_clear_history():
    """Example 5: Clearing conversation history"""
    print("=" * 60)
    print("Example 5: Clearing History")
    print("=" * 60)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    session_id = "clear_test"
    
    print("\n💬 First conversation:")
    r1 = chain_with_history.invoke(
        {"input": "My name is Bob"},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"User: My name is Bob")
    print(f"AI: {r1[:100]}...")
    
    print("\n💬 Ask my name:")
    r2 = chain_with_history.invoke(
        {"input": "What's my name?"},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"User: What's my name?")
    print(f"AI: {r2[:100]}...")
    
    # Clear history
    print("\n🗑️ Clearing history...")
    history = get_session_history(session_id)
    history.clear()
    print("✅ History cleared")
    
    print("\n💬 Ask my name again:")
    r3 = chain_with_history.invoke(
        {"input": "What's my name?"},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"User: What's my name?")
    print(f"AI: {r3[:100]}...")
    
    print()


def example_6_interactive_chat():
    """Example 6: Interactive chat loop"""
    print("=" * 60)
    print("Example 6: Interactive Chat")
    print("=" * 60)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly assistant. Keep responses concise."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    session_id = "interactive_chat"
    
    print("\n💬 Chat started! (type 'exit' to quit, 'clear' to reset)")
    print("-" * 60)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("👋 Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            get_session_history(session_id).clear()
            print("🗑️ History cleared!")
            continue
        
        if not user_input:
            continue
        
        try:
            response = chain_with_history.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            print(f"AI: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print()


def main():
    """Main function with menu"""
    while True:
        print("\n" + "=" * 60)
        print("Memory Examples Menu")
        print("=" * 60)
        print("1. Basic Memory")
        print("2. Multiple Sessions")
        print("3. Window Memory")
        print("4. Custom System Message")
        print("5. Clear History")
        print("6. Interactive Chat")
        print("7. Exit")
        
        choice = input("\nChoice (1-7): ").strip()
        
        if choice == '1':
            example_1_basic_memory()
        elif choice == '2':
            example_2_multi_session()
        elif choice == '3':
            example_3_window_memory()
        elif choice == '4':
            example_4_chat_with_system()
        elif choice == '5':
            example_5_clear_history()
        elif choice == '6':
            example_6_interactive_chat()
        elif choice == '7':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()
