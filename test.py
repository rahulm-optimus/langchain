import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory


load_dotenv()
model = os.getenv("model_2")
llm = OllamaLLM(model=model, temperature=0.7)


def test_prompt_template_partial():
    print("Running test for prompt template with partial variables...\n")
    
    template = PromptTemplate(
        template="{greeting}, {name}!",
        input_variables=["name"],
        partial_variables={"greeting": "Hello"}
    )
    
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            final_prompt = template.format(name=user_input)
            response = llm.invoke(final_prompt)
            print(f"Assistant: {response}\n")
        except Exception as e:
            print(f"An error occurred: {e}. Retrying...")
            

def test_structured_output_parsing():
    print("Running test for structured output parsing...\n ")
    
    parser = JsonOutputParser()

    prompt = PromptTemplate(
        template="Return a summary in JSON with keys 'title' and 'summary'.\n{format_instructions}\nText:{text}",
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    result = chain.invoke({"text": "AI"})
    print(result)
    
def test_chain_composition():
    print("Running test for chain composition...\n LCEL")
    
    prompt1 = PromptTemplate(
        template="Summarize the following text:\n{text}",
        input_variables=["text"]
    )
    
    prompt2 = PromptTemplate(
        template="Translate the following summary to French:\n{summary}",
        input_variables=["summary"]
    )
    
    
    chain = (
        prompt1 
        | llm 
        | StrOutputParser() 
        | prompt2 
        | llm 
        | StrOutputParser()
    )
    
    result = chain.invoke({"text": "AI is transforming the world."})
    print(result)

# stream dataing version
def test_chain_composition():
    print("Running test for chain composition...\n LCEL")
    
    country_info = input("Enter information about a country: ")
    
    prompt1 = PromptTemplate(
        template="Explain about this country in short:\n{text}",
        input_variables=["text"]
    )
    
    prompt2 = PromptTemplate(
        template="What are the places that we can visit in this country :\n{summary}",
        input_variables=["summary"]
    )
    
    
    # LCEL Chain Composition with Streaming
    chain = (
        prompt1 
        | llm 
        | StrOutputParser() 
        | prompt2 
        | llm 
        | StrOutputParser()
    )
    
    # Stream the output in real-time
    print("Streaming response:\n")
    for chunk in chain.stream({f"text": country_info}):
        print(chunk, end="", flush=True)
    print("\n")  # New line after streaming completes
           
 
def test_memory_chain_composition():
     
     print("Running test for chain composition with memory...\n LCEL")
     
     # Store for chat history
     store = {}
     
     def get_session_history(session_id: str) -> BaseChatMessageHistory:
         if session_id not in store:
             store[session_id] = InMemoryChatMessageHistory()
         return store[session_id]
     
     # Create prompt with message history
     prompt = ChatPromptTemplate.from_messages([
         ("system", "You are a helpful AI assistant. Remember the conversation context."),
         MessagesPlaceholder(variable_name="history"),
         ("human", "{input}")
     ])
       
     # Create chain with message history
     chain = prompt | llm | StrOutputParser()
     
     chain_with_history = RunnableWithMessageHistory(
         chain,
         get_session_history,
         input_messages_key="input",
         history_messages_key="history"
     )
     
     print("Chat with memory started! Type 'exit' or 'quit' to end.\n")
     
     while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            response = chain_with_history.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "user_session"}}
            )
            print(f"Assistant: {response}\n")
     
      
def main():
    
    # test_prompt_template_partial()
    # test_structured_output_parsing()
    # test_chain_composition()
    # test_memory_chain_composition()    
            
if __name__ == "__main__":
    main()            