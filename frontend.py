import streamlit as st
import time
import requests
import random
from dotenv import load_dotenv

load_dotenv()


def display_chat_messages() -> None:
    """Print message history
    @returns None
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def process_github_repo(repo_url):
    """Process GitHub repository (dummy function for now)
    @parameter repo_url : str - URL of GitHub repository
    @returns bool - Success status
    """
    # This is a placeholder for the actual API call
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulate processing with steps
    steps = [
        "Cloning repository...",
        "Analyzing code structure...",
        "Processing issues and PRs...",
        "Examining commit history...",
        "Preparing response engine..."
    ]
    
    # Call the API endpoint to process the repository
    response = requests.post("http://localhost:8000/github-chunk", json={"repo_url": repo_url})
    
    # Create illusion of processing - since we don't get real-time status
    for i, step in enumerate(steps):
        # Update status text
        status_text.text(step)
        
        # Update progress bar
        progress_value = (i + 1) / len(steps)
        progress_bar.progress(progress_value)
        
        # Simulate processing time (random between 1-3 seconds per step)
        time.sleep(1)
    
    # Check if the request was successful
    if response.status_code != 200:
        status_text.error(f"Error processing repository: {response.text}")
        time.sleep(2)
        return False
    
    status_text.text("Repository processed successfully!")
    time.sleep(1)
    
    # Clear the progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return True

def get_response(query):
    result = requests.post("http://localhost:8000/github-qa", json={"repo_url": st.session_state.repo_url, "query": query})
    response = result.json()
    return response["answer"]
    
# Title
st.title("RepoMind – GitHub Repository Assistant")

with st.sidebar:
    st.title("🔍 RepoMind")
    st.subheader("Deep Code Intelligence")
    st.markdown(
        """RepoMind is an AI-powered Retrieval-Augmented Generation (RAG) system that provides deep contextual understanding of a GitHub repository, including code, issues, and pull requests. It enables developers to query, analyze, and generate insights based on the entire codebase history, improving collaboration, debugging, and decision-making.
        """
    )
    st.header("Settings")
    st.success("Repository Ready", icon="✅")

# Information
with st.expander("How it works"):
    st.subheader("GitHub Repository Analysis")
    st.markdown(
        """
        This tool analyzes GitHub repositories to provide insights and generate code. It:
        
        1. **Clones the repository** - Downloads the codebase for analysis
        2. **Processes code structure** - Examines files, directories, and dependencies
        3. **Analyzes issues and PRs** - Reviews open issues and pull requests
        4. **Examines commit history** - Studies development patterns over time
        5. **Prepares an AI response engine** - Enables intelligent querying of the repository
        """
    )
    st.subheader("What you can ask")
    st.markdown(
        """You can ask questions about:
- Code architecture and organization
- Open issues and their priorities
- Pull requests and their status
- Performance bottlenecks
- Testing coverage and strategies
- Security concerns

You can also request code generation for specific problems identified in the repository.
        """
    )

# Initialize session state
if "repo_processed" not in st.session_state:
    st.session_state.repo_processed = False
    st.session_state.repo_url = ""
    st.session_state.first_prompt_received = False

# Repository URL input (only shown if not processed)
if not st.session_state.repo_processed:
    repo_url = st.text_input("Enter GitHub Repository URL", placeholder="https://github.com/username/repository")
    
    # Process repository button - disabled if there's no input
    button_disabled = not repo_url  # Button is disabled when repo_url is empty
    if st.button("Process Repository", disabled=button_disabled):
        if st.session_state.repo_url != repo_url or not st.session_state.repo_processed:
            success = process_github_repo(repo_url)
            if success:
                st.session_state.repo_processed = True
                st.session_state.repo_url = repo_url
                
                # Reset chat history when processing a new repo
                if "messages" in st.session_state:
                    st.session_state.messages = []
                    st.session_state.greetings = False
                
                st.rerun()
else:
    # New Repository button in top right when a repo is already processed
    col1, col2 = st.columns([4, 2])
    with col2:
        if st.button("New Repository", use_container_width=True):
            st.session_state.repo_processed = False
            st.session_state.repo_url = ""
            st.session_state.messages = []
            st.session_state.greetings = False
            st.session_state.first_prompt_received = False
            st.rerun()

st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False

# Display chat messages from history on app rerun
display_chat_messages()

# Only show chat interface if repository has been processed
if st.session_state.repo_processed:
    # Greet user
    if not st.session_state.greetings:
        with st.chat_message("assistant"):
            repo_name = st.session_state.repo_url.split("/")[-1] if "/" in st.session_state.repo_url else "repository"
            intro = f"I've analyzed the GitHub repository '{repo_name}'. You can ask me questions about its code structure, issues, PRs, or request code generation based on the repository context. How can I help you today?"
            st.markdown(intro)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": intro})
            st.session_state.greetings = True

    # Show example prompts only if the user hasn't sent a prompt yet
    if not st.session_state.first_prompt_received:
        # Example prompts
        example_prompts = [
            "What's the architecture of this repository?",
            "What are the most critical open issues?",
            "Summarize the open pull requests",
            "Where are the performance bottlenecks?",
            "How's the test coverage for this repo?",
            "Generate code to fix the authentication bug"
        ]

        example_prompts_help = [
            "Analyze code structure and design patterns",
            "Check issues by priority and impact",
            "Review outstanding code contributions",
            "Identify code that could be optimized",
            "Examine test coverage and strategies",
            "Request code generation for fixes"
        ]

        button_cols = st.columns(3)
        button_cols_2 = st.columns(3)

        button_pressed = ""

        if button_cols[0].button(example_prompts[0], help=example_prompts_help[0]):
            button_pressed = example_prompts[0]
        elif button_cols[1].button(example_prompts[1], help=example_prompts_help[1]):
            button_pressed = example_prompts[1]
        elif button_cols[2].button(example_prompts[2], help=example_prompts_help[2]):
            button_pressed = example_prompts[2]
        elif button_cols_2[0].button(example_prompts[3], help=example_prompts_help[3]):
            button_pressed = example_prompts[3]
        elif button_cols_2[1].button(example_prompts[4], help=example_prompts_help[4], use_container_width=True,):
            button_pressed = example_prompts[4]
        elif button_cols_2[2].button(example_prompts[5], help=example_prompts_help[5]):
            button_pressed = example_prompts[5]

    # Handle user input
    if prompt := (st.chat_input("Ask me about the repository...") or button_pressed):
        # Mark that user has sent first prompt
        st.session_state.first_prompt_received = True
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant thinking loader
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            with thinking_placeholder.container():
                with st.spinner("Thinking..."):
                    response = get_response(prompt)
                # Replace the thinking indicator with the actual response
                thinking_placeholder.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
            
else:
    # Show message when no repository has been processed
    st.info("👆 Enter a GitHub repository URL and click 'Process Repository' to begin analysis.")