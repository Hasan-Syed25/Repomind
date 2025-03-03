from typing import Annotated, List, Dict, Any, Optional, Union
from typing_extensions import TypedDict
from datetime import datetime
import time
import asyncio
from langgraph.graph import StateGraph, START, END
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from code_ingestion import *
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
import os

# ----------------- HELPERS ----------------- #

class State(TypedDict):
    messages: Annotated[list, "List of messages"]
    query: Annotated[list, "List of queries"]
    chunker: Annotated[GitHubRepoChunker, "Repository Chunker"]
    nodes: Annotated[list, "List of retrieved context nodes"]
    response: Annotated[list, "List of responses"]
    retries: Annotated[int, "Number of retries"]
    intent: Annotated[str, "Query intent"]
    review_route: Annotated[str, "Route for review decision"]

class CodeReviewResult(BaseModel):
    """Result of code review assessment."""
    is_correct: str = Field(
        description="Whether the code solution is correct, 'yes' or 'no'"
    )
    reasoning: str = Field(
        description="Reasoning behind the assessment"
    )

def format_messages(messages_list: list[dict]) -> str:
    """
    Formats a list of messages into a string with the role as heading and content as text.

    Args:
        messages_list (list): List of dictionaries containing 'role' and 'content' keys.

    Returns:
        str: Formatted string with role as heading and content below.
    """
    formatted_messages = []
    for message in messages_list:
        role = message.get("role", "unknown role").capitalize()
        if role.lower() == "system":
            role = "AI Assistant"
        content = message.get("content", "")
        formatted_messages.append(f"{role}:\n{content}\n")
    joined_messages = "\n".join(formatted_messages)
    formatted_messages = f"## Chat history:\n\n{joined_messages}\n\n-------End of chat history-------"
    return formatted_messages

# Initialize the LLM
AZURE_API_KEY = os.getenv("STRUCTURED_LLM_API_KEY_WE")
llm = AzureChatOpenAI(
        azure_endpoint="https://ai-prodazureaiwesteurope686819533750.openai.azure.com/openai/deployments/prod-gpt4o-mini-westeurope/chat/completions?api-version=2024-08-01-preview",
        model="gpt-4o-mini",
        azure_deployment="prod-gpt4o-mini-westeurope",
        api_key=AZURE_API_KEY,
        api_version="2024-08-01-preview",
        timeout=60
    )

# --------------------------------- LANGCHAIN FUNCTIONS -------------------------------------- #

########## QUERY REFORMULATOR ##########
query_reformulator_system_prompt = """You are a query reformulator specialized in GitHub repositories. 
Your task is to rewrite user queries to optimize them for vector retrieval from GitHub code, issues, and PRs.

Consider:
1. Technical terminology specific to coding and GitHub
2. Code structure and patterns developers typically use
3. Issue/PR naming conventions and discussion formats
4. Expand acronyms and technical shorthand
5. Include likely file types or extensions if implied

Create a search query that would best match the relevant content in a code repository. Do not remove any code blocks fro the query."""

reformulate_prompt = ChatPromptTemplate.from_messages([
    ("system", query_reformulator_system_prompt),
    ("human", "Original query: {query}\n\nPlease reformulate this into an optimized search query for GitHub repository content. Do not remove any code blocks from the original query"),
])

########## CODE REVIEWER ##########
code_reviewer_system_prompt = """You are a code review expert tasked with evaluating the correctness of responses to GitHub repository questions.

Focus on:
1. Technical accuracy of explanations about code, issues, or PRs
2. Correctness of any code solutions or fixes proposed
3. Appropriateness of the answer to the user's question
4. Completeness - whether all aspects of the question were addressed
5. Factual correctness based on the provided repository context

Provide a binary assessment ('yes' or 'no') on whether the response is correct and complete, along with your reasoning."""

code_reviewer_prompt = ChatPromptTemplate.from_messages([
    ("system", code_reviewer_system_prompt),
    ("human", "User question: {question}\n\nRepository context:\n{context}\n\nGenerated answer:\n{answer}\n\nIs this answer correct and complete?"),
])

########## INTENT ANALYZER ##########
intent_analyzer_system_prompt = """You are an intent analyzer for GitHub repository questions. Your task is to determine whether a query requires retrieving additional context from the repository or can be answered directly.

CONTEXT_NEEDED: Queries about specific code, issues, PRs, repository structure, or implementation details that require looking at the actual repository content.

DIRECT_ANSWER: General programming questions, explanations of standard concepts, or queries that don't require specific repository information.

Analyze the query carefully and choose the appropriate category."""

intent_analyzer_prompt = ChatPromptTemplate.from_messages([
    ("system", intent_analyzer_system_prompt),
    ("human", "Query: {query}\n\nDoes this query require retrieving repository context or can it be answered directly?"),
])

# Create chains
reformulator_chain = reformulate_prompt | llm | StrOutputParser()
code_reviewer_chain = code_reviewer_prompt | llm.with_structured_output(CodeReviewResult)
intent_analyzer_chain = intent_analyzer_prompt | llm | StrOutputParser()

# --------------------------------- PROMPT BUILDER -------------------------------------- #

def build_full_prompt(query: str, context: list) -> str:
    """
    Build a complete prompt combining the user query and retrieved context.
    
    Args:
        query (str): User's question
        context (list): Retrieved context from the repository
        
    Returns:
        str: Formatted prompt for the LLM
    """
    context_text = "\n\n".join([f"SOURCE {i+1}:\n{item}" for i, item in enumerate(context)])
    
    prompt = f"""# GitHub Repository Question

## User Query
{query}

## Repository Context
{context_text}

## Instructions
- Answer the user's question based on the repository context provided
- If the context doesn't contain enough information, say so clearly
- For code-related questions, explain the code's purpose and functionality
- For issues/PRs, summarize the key points and status
- Include code examples where appropriate
- Be concise but thorough

Please provide your response:
"""
    return prompt

# --------------------------------- LANGGRAPH FUNCTIONS -------------------------------------- #

async def analyze_intent(state: State):
    """
    Analyze the intent of the query to determine if context retrieval is needed.
    
    Args:
        state (State): Current state
        
    Returns:
        dict: Updated state with intent analysis
    """
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Analyzing query intent...")
    
    query = state["query"][-1]
    
    # Analyze intent using the intent analyzer chain
    intent_result = intent_analyzer_chain.invoke({"query": query})
    
    # Determine the intent based on keywords in the result
    if "CONTEXT_NEEDED" in intent_result:
        intent = "CONTEXT_NEEDED"
    else:
        intent = "DIRECT_ANSWER"
    
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Query intent determined: {intent}")
    
    return {"intent": intent}

async def decide_next_step(state: State) -> str:
    """
    Decides whether to retrieve context or answer directly based on intent analysis.
    
    Args:
        state (State): Current state
        
    Returns:
        str: Next step in the graph
    """
    intent = state.get("intent", "CONTEXT_NEEDED")  # Default to context needed
    
    if intent == "CONTEXT_NEEDED":
        return "RETRIEVE_CONTEXT"
    else:
        return "DIRECT_ANSWER"

async def reformulate_query(state: State):
    """
    Reformulate the query for better retrieval results.
    
    Args:
        state (State): Current state
        
    Returns:
        dict: Updated state with reformulated query
    """
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Reformulating query...")
    
    original_query = state["query"][-1]
    
    # Reformulate the query using the reformulator chain
    reformulated_query = reformulator_chain.invoke({"query": original_query})
    
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Original query: {original_query}")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Reformulated query: {reformulated_query}")
    
    # Add the reformulated query to the state
    state["query"].append(reformulated_query)
    
    return state

async def retrieve_context(state: State):
    """
    Retrieve relevant context from GitHub repository.
    
    Args:
        state (State): Current state
        
    Returns:
        dict: Updated state with retrieved context
    """
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Retrieving context...")
    
    query = state["query"][-1]
    chunker = state["chunker"]
    retries = state["retries"]
    
    # Increment retries counter
    state["retries"] += 1
    
    # Check if maximum retries reached
    if retries >= 2:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Maximum retries reached")
        return state
    
    # Determine the search type based on query content
    if "issues" in query.lower() or "pr" in query.lower() or "pull request" in query.lower():
        search_type = "issues"
    else:
        search_type = None  # Use mixed results
    
    # Search repository
    search_results = chunker.search_repository(
        query_text=query,
        limit=23,
        code_ratio=0.7,  # Adjust based on query type
        only_type=search_type
    )
    
    # Extract context from search results
    context_nodes = []
    for _, result in search_results:
        result_type = result[0].payload.get("type", "unknown")
        
        if result_type == "code_file":
            filename = result[0].payload.get("metadata", {}).get("filename", "unknown")
            content = result[0].payload.get("content", "")
            formatted_content = f"[CODE FILE: {filename}]\n{content}"
            context_nodes.append(formatted_content)
        else:
            issue_type = result[0].payload.get("metadata", {}).get("issue_type", "unknown")
            issue_number = result[0].payload.get("metadata", {}).get("issue_number", "unknown")
            issue_state = result[0].payload.get("metadata", {}).get("issue_state", "unknown")
            content = result[0].payload.get("content", "")
            formatted_content = f"[{issue_type.upper()} #{issue_number} #{issue_state}]\n{content}"
            context_nodes.append(formatted_content)
    
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Retrieved {len(context_nodes)} context nodes")
    
    state["nodes"] = context_nodes
    return state

async def generate_answer(state: State):
    """
    Generate an answer based on the query and retrieved context.
    
    Args:
        state (State): Current state
        
    Returns:
        dict: Updated state with generated answer
    """
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Generating answer...")
    
    query = state["query"][0]  # Use the original query for answer generation
    context = state["nodes"]
    
    if not context:
        response = {
            "answer": "I couldn't find specific information about this in the repository. Could you provide more details or rephrase your question?",
            "sources": []
        }
    else:
        # Build the full prompt
        full_prompt = build_full_prompt(query, context)
        
        # Generate answer using the LLM
        answer = await llm.ainvoke(full_prompt)
        
        response = {
            "answer": answer.content,
            "sources": context
        }
    
    # Add the response to the state
    state["response"].append(response)
    
    return state

async def direct_answer(state: State):
    """
    Generate a direct answer without repository context.
    
    Args:
        state (State): Current state
        
    Returns:
        dict: Updated state with direct answer
    """
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Generating direct answer...")
    
    query = state["query"][0]  # Use the original query
    
    prompt = f"""# GitHub and Programming Question

## User Query
{query}

## Instructions
- Answer this programming or GitHub-related question based on general knowledge
- Provide clear explanations and examples where appropriate
- Be concise but thorough

Please provide your response:
"""
    
    # Generate answer using the LLM
    answer = await llm.ainvoke(prompt)
    
    response = {
        "answer": answer.content,
        "sources": []
    }
    
    # Add the response to the state
    state["response"].append(response)
    
    return state

async def review_code(state: State):
    """
    Review the generated answer for correctness.
    
    Args:
        state (State): Current state
        
    Returns:
        str: Decision based on review
    """
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Reviewing answer...")
    
    query = state["query"][0]  # Original query
    context = state["nodes"]
    answer = state["response"][-1]["answer"]
    
    # If no context was used (direct answer), assume it's correct
    if not context:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} No context used, skipping review")
        return {"review_route": "CORRECT"}
    
    # Format context for review
    context_text = "\n\n".join([f"SOURCE {i+1}:\n{item}\n" for i, item in enumerate(context)])
    
    # Review the code using the code reviewer chain
    review_result = code_reviewer_chain.invoke({
        "question": query,
        "context": context_text,
        "answer": answer
    })
    
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Review result: {review_result.is_correct}")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Review reasoning: {review_result.reasoning}")
    
    
    if review_result.is_correct.lower() == "yes":
        return {"review_route": "CORRECT"}
    else:
        # Only allow one retry
        if state["retries"] >= 2:
            return {"review_route": "CORRECT"} # Return the best answer we have
        else:
            return {"review_route": "INCORRECT"} 
        

# --------------------------------- GRAPH DEFINITION -------------------------------------- #

# Build the graph
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("ANALYZE_INTENT", analyze_intent)
graph_builder.add_node("REFORMULATE_QUERY", reformulate_query)
graph_builder.add_node("RETRIEVE_CONTEXT", retrieve_context)
graph_builder.add_node("GENERATE_ANSWER", generate_answer)
graph_builder.add_node("DIRECT_ANSWER", direct_answer)
graph_builder.add_node("REVIEW_CODE", review_code)

# Add edges
graph_builder.add_edge(START, "ANALYZE_INTENT")
graph_builder.add_conditional_edges(
    "ANALYZE_INTENT",
    decide_next_step,
    {
        "RETRIEVE_CONTEXT": "REFORMULATE_QUERY",
        "DIRECT_ANSWER": "DIRECT_ANSWER"
    }
)
graph_builder.add_edge("REFORMULATE_QUERY", "RETRIEVE_CONTEXT")
graph_builder.add_edge("RETRIEVE_CONTEXT", "GENERATE_ANSWER")
graph_builder.add_edge("DIRECT_ANSWER", "REVIEW_CODE")
graph_builder.add_edge("GENERATE_ANSWER", "REVIEW_CODE")
graph_builder.add_conditional_edges(
    "REVIEW_CODE",
    lambda state: state["review_route"],
    {
        "CORRECT": END,
        "INCORRECT": "REFORMULATE_QUERY"
    }
)

# Compile the graph
graph = graph_builder.compile()

# --------------------------------- MAIN FUNCTION -------------------------------------- #

async def github_qa_agent(chunker, query, chat_history=None):
    """
    Main function to answer GitHub repository questions.
    
    Args:
        client: Chunker Class instance
        query (str): User query
        chat_history (list, optional): Previous chat history
        
    Returns:
        dict: Response with answer and sources
    """
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Processing query: {query}")
    
    # Format chat history if provided
    messages = []
    if chat_history:
        messages = chat_history[-5:]  # Use last 5 messages
    
    # Set initial state
    state = {
        "messages": messages,
        "query": [query],
        "chunker": chunker,
        "nodes": [],
        "response": [],
        "retries": 0,
        "intent": "",
        "review_route": ""
    }
    
    # Run the graph
    result = await graph.ainvoke(state)
    
    # Extract the final response
    last_response = result.get("response", [])
    if last_response:
        response = {
            "answer": last_response[-1]["answer"],
            "sources": last_response[-1]["sources"]
        }
    else:
        response = {
            "answer": "I couldn't process your question. Could you try rephrasing it?",
            "sources": []
        }
    
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Query processing complete")
    
    return response

# Example usage:
async def main():
    chunker = GitHubRepoChunker(
        repo_url="https://github.com/Hasan-Syed25/HF2Reasoning",
        collection_name="test"
    )
    
    response = await github_qa_agent(chunker, "How does the authentication system work in this repo?")
    print(response["answer"])

if __name__ == "__main__":
    asyncio.run(main())
