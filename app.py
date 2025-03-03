from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from datetime import datetime
from agent import GitHubRepoChunker, github_qa_agent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the request model
class QueryRequest(BaseModel):
    repo_url: str
    query: str
    chat_history: list = None
    
class ChunkRequest(BaseModel):
    repo_url: str

# Initialize FastAPI app
app = FastAPI()

# Define the endpoint
@app.post("/github-qa")
async def github_qa_endpoint(request: QueryRequest):
    """
    FastAPI endpoint to answer questions about a GitHub repository.
    
    Args:
        request (QueryRequest): Contains repo_url, query, and optional chat_history
        
    Returns:
        dict: Response with answer and sources
        
    Raises:
        HTTPException: If an error occurs during processing
    """
    logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Received request for repo: {request.repo_url}, query: {request.query}")
    
    try:
        # Initialize the GitHubRepoChunker
        chunker = GitHubRepoChunker(
            repo_url=request.repo_url,
            collection_name="test"  # Hardcoded for simplicity; could be made dynamic
        )
        logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Chunker initialized for repo: {request.repo_url}")
        
        # Call the github_qa_agent function with the provided query and optional chat history
        response = await github_qa_agent(chunker, request.query, request.chat_history)
        logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Response generated for query: {request.query}")
        
        return response
    except Exception as e:
        logger.error(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Another Endpoint for Just Chunking the Repo

@app.post("/github-chunk")
async def github_chunk_endpoint(request: ChunkRequest):
    """
    FastAPI endpoint to chunk a GitHub repository.
    
    Args:
        request (QueryRequest): Contains repo_url
        
    Returns:
        dict: Response with chunked data
        
    Raises:
        HTTPException: If an error occurs during processing
    """
    logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Received request to chunk repo: {request.repo_url}")
    
    try:
        # Initialize the GitHubRepoChunker
        chunker = GitHubRepoChunker(
            repo_url=request.repo_url,
            collection_name="test"  # Hardcoded for simplicity; could be made dynamic
        )
        logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Chunker initialized for repo: {request.repo_url}")
        
        # Call the chunker method to chunk the repo
        results = await chunker.process_repository()
    
        print("\nRepository Processing Summary:")
        print(f"Total code files processed: {results['code_files_count']}")
        print(f"Total issues/PRs processed: {results['issues_prs_count']}")
        print(f"Total chunks stored in Qdrant: {results['total_chunks']}")
        print(f"Qdrant collection name: {results['collection_name']}")
        logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Repo chunked successfully")
        
        return results
    except Exception as e:
        logger.error(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")