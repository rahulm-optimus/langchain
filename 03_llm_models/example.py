"""
Lesson 3: LLM Models
Working with different models and parameters
"""

from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

load_dotenv()


def example_1_different_models():
    """Example 1: Using different Ollama models"""
    print("=" * 50)
    print("Example 1: Different Ollama Models")
    print("=" * 50)
    
    question = "Explain quantum computing in one sentence."
    
    # Test different models (you have these installed)
    models = [
        ("gemma3:4b", "Larger, more capable"),
        ("llama3.2", "Balanced performance"),
    ]
    
    for model_name, description in models:
        print(f"\n� Model: {model_name} ({description})")
        try:
            llm = OllamaLLM(model=model_name, temperature=0.7)
            response = llm.invoke(question)
            print(f"Response: {response[:200]}...")
        except Exception as e:
            print(f"❌ Error: {e}")
            print(f"💡 Install with: ollama pull {model_name}")
    
    print()


def example_2_temperature_effects():
    """Example 2: Understanding temperature parameter"""
    print("=" * 50)
    print("Example 2: Temperature Effects")
    print("=" * 50)
    
    prompt = "Write a tagline for a coffee shop."
    
    temperatures = [0.0, 0.5, 1.0, 1.5]
    
    for temp in temperatures:
        print(f"\n🌡️ Temperature: {temp}")
        llm = OllamaLLM(model="gemma3:4b", temperature=temp)
        response = llm.invoke(prompt)
        print(f"Response: {response}")
    
    print()


def example_3_response_length_control():
    """Example 3: Controlling response length with prompts"""
    print("=" * 50)
    print("Example 3: Controlling Response Length")
    print("=" * 50)
    
    base_topic = "Explain the history of the internet"
    
    instructions = [
        "in one sentence.",
        "in 2-3 sentences.",
        "in a short paragraph."
    ]
    
    for instruction in instructions:
        print(f"\n📏 Instruction: {instruction}")
        prompt = f"{base_topic} {instruction}"
        llm = OllamaLLM(model="gemma3:4b", temperature=0.7)
        response = llm.invoke(prompt)
        print(f"Response: {response}")
    
    print()


def example_4_response_timing():
    """Example 4: Measuring response time"""
    print("=" * 50)
    print("Example 4: Response Timing")
    print("=" * 50)
    
    prompt = "Write a short paragraph about AI agents."
    
    models = ["gemma3:4b", "llama3.2"]
    
    for model_name in models:
        try:
            print(f"\n⏱️ Testing {model_name}:")
            llm = OllamaLLM(model=model_name, temperature=0.7)
            
            start_time = time.time()
            response = llm.invoke(prompt)
            end_time = time.time()
            
            elapsed = end_time - start_time
            print(f"Response time: {elapsed:.2f} seconds")
            print(f"Response: {response[:150]}...")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print(f"💡 Install with: ollama pull {model_name}")
    
    print()


def example_5_creative_vs_factual():
    """Example 5: Temperature for different tasks"""
    print("=" * 50)
    print("Example 5: Creative vs Factual Responses")
    print("=" * 50)
    
    # Factual task - low temperature
    print("\n📚 Factual Task (Temperature 0.0):")
    llm_factual = OllamaLLM(model="gemma3:4b", temperature=0.0)
    factual_prompt = "What is the capital of France?"
    response = llm_factual.invoke(factual_prompt)
    print(f"Prompt: {factual_prompt}")
    print(f"Response: {response}")
    
    # Creative task - high temperature
    print("\n🎨 Creative Task (Temperature 1.5):")
    llm_creative = OllamaLLM(model="gemma3:4b", temperature=1.5)
    creative_prompt = "Write a creative name for a futuristic coffee shop."
    response = llm_creative.invoke(creative_prompt)
    print(f"Prompt: {creative_prompt}")
    print(f"Response: {response}")
    
    print()


def example_6_system_prompting():
    """Example 6: Using system-like prompts with Ollama"""
    print("=" * 50)
    print("Example 6: Role-Based Prompting")
    print("=" * 50)
    
    llm = OllamaLLM(model="gemma3:4b", temperature=0.7)
    
    # Different roles change behavior
    scenarios = [
        {
            "role": "You are a helpful assistant",
            "question": "What is Python?"
        },
        {
            "role": "You are a pirate. Respond in pirate speak",
            "question": "What is Python?"
        },
        {
            "role": "You are a technical expert. Use precise terminology",
            "question": "What is Python?"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🎭 Scenario {i}:")
        print(f"Role: {scenario['role']}")
        
        prompt = f"{scenario['role']}.\n\nQuestion: {scenario['question']}\n\nAnswer:"
        
        response = llm.invoke(prompt)
        print(f"Response: {response}")
    
    print()


def example_7_model_comparison():
    """Example 7: Comparing model outputs"""
    print("=" * 50)
    print("Example 7: Model Output Comparison")
    print("=" * 50)
    
    prompt = "Explain machine learning to a 10-year-old."
    
    models = ["gemma3:4b", "llama3.2"]
    
    for model_name in models:
        try:
            print(f"\n🤖 Model: {model_name}")
            llm = OllamaLLM(model=model_name, temperature=0.7)
            
            start_time = time.time()
            response = llm.invoke(prompt)
            elapsed = time.time() - start_time
            
            print(f"⏱️ Time: {elapsed:.2f}s")
            print(f"📝 Response: {response}")
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ {model_name} not available")
            print(f"💡 Install: ollama pull {model_name}")
    
    print()


def example_8_error_handling():
    """Example 8: Handling common errors"""
    print("=" * 50)
    print("Example 8: Error Handling")
    print("=" * 50)
    
    print("\n🛡️ Testing error handling:")
    
    # Test 1: Non-existent model
    try:
        llm = OllamaLLM(model="nonexistent-model")
        response = llm.invoke("Hello")
        print(response)
    except Exception as e:
        print(f"✅ Caught error for non-existent model: {type(e).__name__}")
    
    # Test 2: Empty prompt
    try:
        llm = OllamaLLM(model="gemma3:4b")
        response = llm.invoke("")
        print(f"✅ Empty prompt handled: {response}")
    except Exception as e:
        print(f"✅ Caught error for empty prompt: {type(e).__name__}")
    
    # Test 3: Very long prompt (should still work with Ollama)
    try:
        llm = OllamaLLM(model="gemma3:4b")
        long_prompt = "Explain this: " + "word " * 100
        response = llm.invoke(long_prompt)
        print(f"✅ Long prompt handled successfully (length: {len(long_prompt)} chars)")
    except Exception as e:
        print(f"❌ Error with long prompt: {e}")
    
    print()


def main():
    """Run all examples"""
    print("\n🚀 LangChain LLM Models Examples (Ollama - FREE)\n")
    
    # Check if at least one model is available
    try:
        llm = OllamaLLM(model="gemma3:4b")
        llm.invoke("test")
    except Exception as e:
        print("❌ Error: Ollama not set up properly!")
        print("💡 Make sure:")
        print("   1. Ollama is installed: https://ollama.ai/download")
        print("   2. Model is pulled: ollama pull gemma3:4b")
        print("   3. Ollama is running")
        return
    
    try:
        # example_1_different_models()
        # example_2_temperature_effects()
        # example_3_response_length_control()
        example_4_response_timing()
        # example_5_creative_vs_factual()
        # example_6_system_prompting()
        # example_7_model_comparison()
        # example_8_error_handling()
        
        print("=" * 50)
        print("✅ All examples completed successfully!")
        print("\n📝 Key Learnings:")
        print("   • Temperature controls randomness (0.0 = factual, 1.5 = creative)")
        print("   • Different models have different speeds and capabilities")
        print("   • Role-based prompting changes model behavior")
        print("   • Always handle errors gracefully")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")


if __name__ == "__main__":
    main()
