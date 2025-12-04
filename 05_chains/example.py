"""
Lesson 5: Chains Examples
"""

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser


# ============================================================================
# PART 1: LCEL - MODERN APPROACH (RECOMMENDED)
# ============================================================================

def example_1_basic_lcel():
    """Basic LCEL chain with prompt | llm | parser"""
    print("\n=== Example 1: Basic LCEL Chain ===")
    
    prompt = PromptTemplate(
        template="Tell me a short joke about {topic}",
        input_variables=["topic"]
    )
    
    llm = OllamaLLM(model="llama3.2", temperature=0.7)
    
    # LCEL: Chain with pipe operator
    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({"topic": "programming"})
    print(f"Result: {result}")


def example_2_multi_step_lcel():
    """Multi-step LCEL chain"""
    print("\n=== Example 2: Multi-Step LCEL Chain ===")
    
    prompt1 = PromptTemplate(
        template="Summarize this in one sentence: {text}",
        input_variables=["text"]
    )
    
    prompt2 = PromptTemplate(
        template="Translate to Spanish: {summary}",
        input_variables=["summary"]
    )
    
    llm = OllamaLLM(model="llama3.2", temperature=0.7)
    
    # Two-step chain
    chain = (
        prompt1 
        | llm 
        | StrOutputParser() 
        | (lambda summary: {"summary": summary})  # Convert to dict for next prompt
        | prompt2 
        | llm 
        | StrOutputParser()
    )
    
    result = chain.invoke({"text": "Python is a versatile programming language."})
    print(f"Result: {result}")


def example_3_lcel_with_streaming():
    """LCEL chain with streaming support"""
    print("\n=== Example 3: LCEL with Streaming ===")
    
    prompt = PromptTemplate(
        template="Write a short story about {topic}",
        input_variables=["topic"]
    )
    
    llm = OllamaLLM(model="llama3.2", temperature=0.7)
    chain = prompt | llm | StrOutputParser()
    
    print("Streaming response:")
    for chunk in chain.stream({"topic": "a robot learning to feel emotions"}):
        print(chunk, end="", flush=True)
    print("\n")


def example_4_lcel_with_json_parser():
    """LCEL with JSON output"""
    print("\n=== Example 4: LCEL with JSON Parser ===")
    
    parser = JsonOutputParser()
    
    prompt = PromptTemplate(
        template="""Generate a person profile in JSON format.
Name: {name}
{format_instructions}""",
        input_variables=["name"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    llm = OllamaLLM(model="llama3.2", temperature=0.7)
    chain = prompt | llm | parser
    
    result = chain.invoke({"name": "Alice"})
    print(f"Result: {result}")


# ============================================================================
# PART 2: LEGACY CHAINS (Still Work)
# ============================================================================

def example_5_llm_chain():
    """Basic LLMChain (legacy approach)"""
    print("\n=== Example 5: LLMChain (Legacy) ===")
    
    prompt = PromptTemplate(
        template="What is {topic}?",
        input_variables=["topic"]
    )
    
    llm = OllamaLLM(model="llama3.2", temperature=0.7)
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    result = chain.invoke({"topic": "machine learning"})
    print(f"Result: {result}")


def example_6_simple_sequential_chain():
    """SimpleSequentialChain - single variable flow"""
    print("\n=== Example 6: SimpleSequentialChain ===")
    
    llm = OllamaLLM(model="llama3.2", temperature=0.7)
    
    # Chain 1: Generate a topic
    prompt1 = PromptTemplate(
        template="Suggest one interesting topic about {subject}",
        input_variables=["subject"]
    )
    chain1 = LLMChain(llm=llm, prompt=prompt1)
    
    # Chain 2: Write about the topic
    prompt2 = PromptTemplate(
        template="Write two sentences about: {topic}",
        input_variables=["topic"]
    )
    chain2 = LLMChain(llm=llm, prompt=prompt2)
    
    # Combine chains
    overall_chain = SimpleSequentialChain(
        chains=[chain1, chain2],
        verbose=True
    )
    
    result = overall_chain.invoke("artificial intelligence")
    print(f"Final Result: {result}")


def example_7_sequential_chain():
    """SequentialChain - multiple variables"""
    print("\n=== Example 7: SequentialChain ===")
    
    llm = OllamaLLM(model="llama3.2", temperature=0.7)
    
    # Chain 1: Generate story
    prompt1 = PromptTemplate(
        template="Write a very short story about {topic}",
        input_variables=["topic"]
    )
    chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="story")
    
    # Chain 2: Summarize story
    prompt2 = PromptTemplate(
        template="Summarize this story in one line: {story}",
        input_variables=["story"]
    )
    chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="summary")
    
    # Combine with multiple outputs
    overall_chain = SequentialChain(
        chains=[chain1, chain2],
        input_variables=["topic"],
        output_variables=["story", "summary"],
        verbose=True
    )
    
    result = overall_chain.invoke({"topic": "a time traveler"})
    print(f"Story: {result['story']}")
    print(f"Summary: {result['summary']}")


# ============================================================================
# PART 3: TRANSFORM CHAINS
# ============================================================================

def example_8_transform_chain():
    """TransformChain - apply custom Python functions"""
    print("\n=== Example 8: TransformChain ===")
    
    def transform_func(inputs: dict) -> dict:
        """Custom transformation: convert to uppercase and count words"""
        text = inputs["text"]
        upper_text = text.upper()
        word_count = len(text.split())
        return {
            "output_text": upper_text,
            "word_count": word_count
        }
    
    transform_chain = TransformChain(
        input_variables=["text"],
        output_variables=["output_text", "word_count"],
        transform=transform_func
    )
    
    result = transform_chain.invoke({"text": "Hello world from Python"})
    print(f"Transformed: {result['output_text']}")
    print(f"Word Count: {result['word_count']}")


def example_9_transform_with_llm():
    """Combine TransformChain with LLM"""
    print("\n=== Example 9: TransformChain + LLM ===")
    
    llm = OllamaLLM(model="llama3.2", temperature=0.7)
    
    # Transform function
    def clean_text(inputs: dict) -> dict:
        text = inputs["text"].strip().lower()
        return {"cleaned_text": text}
    
    transform_chain = TransformChain(
        input_variables=["text"],
        output_variables=["cleaned_text"],
        transform=clean_text
    )
    
    # LLM chain
    prompt = PromptTemplate(
        template="Analyze this text: {cleaned_text}",
        input_variables=["cleaned_text"]
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt, output_key="analysis")
    
    # Combine
    overall_chain = SequentialChain(
        chains=[transform_chain, llm_chain],
        input_variables=["text"],
        output_variables=["analysis"]
    )
    
    result = overall_chain.invoke({"text": "  PYTHON IS AWESOME!  "})
    print(f"Analysis: {result['analysis']}")


# ============================================================================
# PART 4: CONVERSATION CHAIN WITH MEMORY
# ============================================================================

def example_10_conversation_chain():
    """ConversationChain with memory"""
    print("\n=== Example 10: ConversationChain with Memory ===")
    
    llm = OllamaLLM(model="llama3.2", temperature=0.7)
    
    # Initialize memory
    memory = ConversationBufferMemory()
    
    # Create conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    # First interaction
    response1 = conversation.invoke("Hi, my name is Alice")
    print(f"Response 1: {response1}\n")
    
    # Second interaction - remembers context
    response2 = conversation.invoke("What's my name?")
    print(f"Response 2: {response2}\n")
    
    # Third interaction
    response3 = conversation.invoke("What did we talk about?")
    print(f"Response 3: {response3}")


# ============================================================================
# PART 5: CUSTOM LCEL PATTERNS
# ============================================================================

def example_11_lcel_with_custom_logic():
    """LCEL with custom Python functions inline"""
    print("\n=== Example 11: LCEL with Custom Logic ===")
    
    prompt = PromptTemplate(
        template="List 3 {category}",
        input_variables=["category"]
    )
    
    llm = OllamaLLM(model="llama3.2", temperature=0.7)
    
    # Custom function to process output
    def format_list(text: str) -> dict:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return {"formatted": "\n".join(f"✓ {line}" for line in lines[:3])}
    
    chain = (
        prompt 
        | llm 
        | StrOutputParser() 
        | format_list
    )
    
    result = chain.invoke({"category": "programming languages"})
    print(f"Formatted List:\n{result['formatted']}")


def example_12_lcel_parallel_chains():
    """Run multiple chains and combine results"""
    print("\n=== Example 12: LCEL Parallel Chains ===")
    
    from langchain_core.runnables import RunnableParallel
    
    llm = OllamaLLM(model="llama3.2", temperature=0.7)
    
    # Define prompts
    prompt1 = PromptTemplate(
        template="What are pros of {topic}?",
        input_variables=["topic"]
    )
    
    prompt2 = PromptTemplate(
        template="What are cons of {topic}?",
        input_variables=["topic"]
    )
    
    # Create parallel chains
    chain = RunnableParallel({
        "pros": prompt1 | llm | StrOutputParser(),
        "cons": prompt2 | llm | StrOutputParser()
    })
    
    result = chain.invoke({"topic": "remote work"})
    print(f"Pros: {result['pros']}\n")
    print(f"Cons: {result['cons']}")


# ============================================================================
# PART 6: REAL-WORLD EXAMPLES
# ============================================================================

def example_13_content_pipeline():
    """Real-world: Content creation pipeline"""
    print("\n=== Example 13: Content Creation Pipeline ===")
    
    llm = OllamaLLM(model="llama3.2", temperature=0.7)
    
    # Step 1: Generate outline
    outline_prompt = PromptTemplate(
        template="Create a brief outline for an article about {topic}",
        input_variables=["topic"]
    )
    
    # Step 2: Write content
    content_prompt = PromptTemplate(
        template="Write a short article based on this outline:\n{outline}",
        input_variables=["outline"]
    )
    
    # Step 3: Add title
    title_prompt = PromptTemplate(
        template="Generate a catchy title for this article:\n{article}",
        input_variables=["article"]
    )
    
    # Build pipeline
    pipeline = (
        outline_prompt
        | llm
        | StrOutputParser()
        | (lambda outline: {"outline": outline})
        | content_prompt
        | llm
        | StrOutputParser()
        | (lambda article: {"article": article})
        | title_prompt
        | llm
        | StrOutputParser()
    )
    
    result = pipeline.invoke({"topic": "benefits of learning Python"})
    print(f"Generated Title: {result}")


def example_14_analysis_workflow():
    """Real-world: Data analysis workflow"""
    print("\n=== Example 14: Analysis Workflow ===")
    
    llm = OllamaLLM(model="llama3.2", temperature=0.7)
    
    # Simulate data cleaning
    def clean_data(inputs: dict) -> dict:
        data = inputs["raw_data"].strip().replace("  ", " ")
        return {"cleaned_data": data}
    
    clean_chain = TransformChain(
        input_variables=["raw_data"],
        output_variables=["cleaned_data"],
        transform=clean_data
    )
    
    # Analyze
    analysis_prompt = PromptTemplate(
        template="Analyze this data and provide insights: {cleaned_data}",
        input_variables=["cleaned_data"]
    )
    analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt, output_key="analysis")
    
    # Summarize
    summary_prompt = PromptTemplate(
        template="Summarize these insights in one sentence: {analysis}",
        input_variables=["analysis"]
    )
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="summary")
    
    # Complete workflow
    workflow = SequentialChain(
        chains=[clean_chain, analysis_chain, summary_chain],
        input_variables=["raw_data"],
        output_variables=["cleaned_data", "analysis", "summary"]
    )
    
    result = workflow.invoke({"raw_data": "  Sales increased by 25% in Q3  "})
    print(f"Summary: {result['summary']}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all examples"""
    print("=" * 70)
    print("LANGCHAIN CHAINS - COMPLETE EXAMPLES")
    print("=" * 70)
    
    # Uncomment the examples you want to run
    
    # LCEL - Modern Approach
    example_1_basic_lcel()
    # example_2_multi_step_lcel()
    # example_3_lcel_with_streaming()
    # example_4_lcel_with_json_parser()
    
    # Legacy Chains
    # example_5_llm_chain()
    # example_6_simple_sequential_chain()
    # example_7_sequential_chain()
    
    # Transform Chains
    # example_8_transform_chain()
    # example_9_transform_with_llm()
    
    # Memory & Conversation
    # example_10_conversation_chain()
    
    # Advanced LCEL
    # example_11_lcel_with_custom_logic()
    # example_12_lcel_parallel_chains()
    
    # Real-world Examples
    # example_13_content_pipeline()
    # example_14_analysis_workflow()
    
    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
