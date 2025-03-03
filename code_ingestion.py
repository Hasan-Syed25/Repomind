import os
import asyncio
import subprocess
import glob
import re
from typing import List, Dict, Any
import uuid
import time

# For repository fetching
from gitingest import ingest_async

# For vector embeddings
from fastembed import SparseTextEmbedding, TextEmbedding

# For vector storage
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Prefetch, FusionQuery, Fusion
import shutil
from dotenv import load_dotenv
load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_API_URL = os.getenv("QDRANT_API_URL")

class GitHubRepoChunker:
    def __init__(self, repo_url: str, collection_name: str):
        """
        Initialize the GitHubRepoChunker.
        
        Args:
            repo_url: The URL of the GitHub repository to fetch
            collection_name: Name for the Qdrant collection
        """
        self.repo_url = repo_url
        self.repo_name = repo_url.split('/')[-1]
        self.owner = repo_url.split('/')[-2]
        self.collection_name = collection_name
        
        # Initialize embedding models
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        self.dense_model = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-en")
        self.qdrant_client = QdrantClient(url=QDRANT_API_URL, port=6333, api_key=QDRANT_API_KEY, timeout=2000)


    async def fetch_repository(self):
        """Fetches the entire repository content using gitingest."""
        print(f"Fetching repository content from {self.repo_url}...")
        time.sleep(1)
        self.summary, self.tree, self.content = await ingest_async(self.repo_url)
        print(f"Repository content fetched successfully.")
        return self.summary, self.tree, self.content

    def fetch_issues_and_prs(self):
        """Fetches issues and PRs using gh2md."""
        print(f"Fetching issues and PRs for {self.owner}/{self.repo_name}...")
        output_dir = "github-repo"
        
        # Check if gh CLI is installed
        try:
            subprocess.run(["gh", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("GitHub CLI (gh) is not installed or not properly configured.")
            print("Please install it from: https://cli.github.com/")
            return False

        # Run gh2md to fetch issues and PRs
        try:
            subprocess.run(["gh2md", f"{self.owner}/{self.repo_name}", output_dir, "--multiple-files"])
            print(f"Issues and PRs fetched successfully to {output_dir}/")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error fetching issues and PRs: {e}")
            print("Make sure gh2md is installed: pip install gh2md")
            return False

    def parse_code_files(self):
        """
        Parse code files from the repository content.
        
        Returns:
            List of dictionaries with file info and content
        """
        chunks = []
        # Updated regex pattern to match the actual format of the files
        # More robust pattern to extract filename and content
        file_pattern = re.compile(r"File: (.*?)\n(.*?)(?=\nFile:|$)", re.DOTALL)
        
        for match in file_pattern.finditer(self.content):
            filename = match.group(1)
            filename = filename.replace("=", "")
            content = match.group(2)
            content = content.strip("=")
            
            chunks.append({
                "type": "code_file",
                "filename": filename,
                "content": content,
                "summary": self.summary,
                "tree_location": self.find_file_in_tree(filename),
            })
        
        print(f"Parsed {len(chunks)} code files")
        return chunks

    def find_file_in_tree(self, filename):
        """Find a file's path in the tree structure."""
        file_parts = filename.split('/')
        
        def search_tree(tree_part, path_so_far, remaining_parts):
            if not remaining_parts:
                return path_so_far
            
            current_part = remaining_parts[0]
            
            for item in tree_part:
                if isinstance(item, dict) and item.get('name') == current_part:
                    new_path = f"{path_so_far}/{current_part}"
                    if 'children' in item and len(remaining_parts) > 1:
                        result = search_tree(item['children'], new_path, remaining_parts[1:])
                        if result:
                            return result
                    else:
                        return new_path
            
            return None
        
        return search_tree(self.tree, "", file_parts)

    def parse_issues_and_prs(self):
        """
        Parse issues and PRs from the github-repo directory.
        
        Returns:
            List of dictionaries with issue/PR info and content
        """
        chunks = []
        output_dir = "github-repo"
        
        if not os.path.exists(output_dir):
            print(f"Directory {output_dir} not found. Issues and PRs might not have been fetched.")
            return chunks
        
        md_files = glob.glob(f"{output_dir}/*.md")
        
        for file_path in md_files:
            file_name = os.path.basename(file_path)
            file_parts = file_name.split('.')
            if len(file_parts) >= 4:
                created_at = file_parts[0]
                issue_number = file_parts[1]
                issue_type = file_parts[2]
                issue_state = file_parts[3].replace('.md', '')
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                chunks.append({
                    "type": "issue_pr",
                    "issue_type": issue_type,
                    "issue_number": issue_number,
                    "issue_state": issue_state,
                    "created_at": created_at,
                    "content": content,
                    "summary": self.summary,
                    "tree_info": None,  # Issues don't have tree location
                })
        
        print(f"Parsed {len(chunks)} issues and PRs")
        return chunks

    def get_embeddings(self, text: str):
        """
        Get both sparse and dense embeddings using FastEmbed.
        
        Args:
            text: The text to embed
            
        Returns:
            Tuple of (sparse_embedding, dense_embedding)
        """
        if len(text) > 8000:
            text = text[:8000]
        
        try:
            sparse_embedding = list(self.sparse_model.query_embed(text))[0]
        except Exception as e:
            print(f"Error getting sparse embedding: {e}")
            # Return empty sparse vector as fallback
            sparse_embedding = {"indices": [], "values": []}
        
        # Get dense embedding (Jina)
        try:
            dense_embedding = list(self.dense_model.query_embed(text))[0]
        except Exception as e:
            print(f"Error getting dense embedding: {e}")
            # Return zero vector as fallback
            dense_embedding = [0.0] * 768  # Adjust size based on model used
        
        return sparse_embedding, dense_embedding

    def setup_qdrant_collection(self):
        """
        Set up the Qdrant collection. Delete if exists and recreate.
        """
        # Check if collection exists
        collections = self.qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name in collection_names:
            print(f"Collection {self.collection_name} already exists. Deleting...")
            self.qdrant_client.delete_collection(collection_name=self.collection_name)
        
        # Create new collection with hybrid search capability
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "jina": models.VectorParams(
                    size=768,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "bm42": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            }
        )
        print(f"Created new collection: {self.collection_name}")

    def store_chunks_in_qdrant(self, chunks: List[Dict[str, Any]]):
        """
        Store all chunks in Qdrant with both sparse and dense vectors.
        
        Args:
            chunks: List of chunk dictionaries
        """
        # Prepare points for batch upload
        points = []
        
        for i, chunk in enumerate(chunks):
            # Get embeddings
            sparse_embedding, dense_embedding = self.get_embeddings(chunk["content"])
            
            # Create point
            point = models.PointStruct(
                id=uuid.uuid4().hex,
                vector={
                    "bm42": sparse_embedding.as_object(),
                    "jina": dense_embedding.tolist()
                },
                payload={
                    "type": chunk["type"],
                    "content": chunk["content"],
                    "summary": chunk["summary"],
                    "metadata": {
                        "tree_info": chunk.get("tree_location"),
                        "filename": chunk.get("filename", ""),
                        "issue_type": chunk.get("issue_type", ""),
                        "issue_number": chunk.get("issue_number", ""),
                        "issue_state": chunk.get("issue_state", ""),
                        "created_at": chunk.get("created_at", "")
                    }
                }
            )
            points.append(point)
            
            # Upload in batches of 100 to avoid timeouts
            if len(points) >= 100 or i == len(chunks) - 1:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"Uploaded batch of {len(points)} points to Qdrant")
                points = []
        
        print(f"Successfully stored {len(chunks)} chunks in Qdrant collection {self.collection_name}")

    def search_repository(self, query_text: str, limit: int = 10, code_ratio: float = 0.7, only_type: str = None):
        """
        Advanced repository search using hybrid search with contextual boosting.
        
        This implementation:
        1. Performs initial hybrid search across all content
        2. Intelligently balances code files vs issues/PRs based on query content
        3. Uses keyword analysis to detect code-specific queries
        4. Performs query expansion for issues/PRs to find related code
        5. Uses reciprocal rank fusion for optimal result blending
        
        Args:
            query_text: The search query
            limit: Maximum number of results to return
            code_ratio: Default ratio of code files vs issues/PRs (0-1)
            only_type: If "code", return only code files; if "issues", return only issues/PRs; 
                       if None, return mixed results based on code_ratio
                
        Returns:
            List of (score, result) tuples sorted by relevance
        """
        # Get embeddings for the query
        sparse_embedding, dense_embedding = self.get_embeddings(query_text)
        
        # Set up initial filter based on only_type parameter
        filter_condition = None
        if only_type == "code":
            filter_condition = models.Filter(
                must=[models.FieldCondition(key="type", match=models.MatchValue(value="code_file"))]
            )
        elif only_type == "issues":
            filter_condition = models.Filter(
                must=[models.FieldCondition(key="type", match=models.MatchValue(value="issue_pr"))]
            )
        
        # Initial search with extended limit to allow for filtering
        extended_limit = limit * 3
        search_result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(query=sparse_embedding.as_object(), using="bm42", limit=extended_limit),
                Prefetch(query=dense_embedding.tolist(), using="jina", limit=extended_limit),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=extended_limit,
            query_filter=filter_condition
        )
        
        # If filtering by type, just return the sorted results
        if only_type:
            results = []
            for scored_point in search_result.points:
                results.append((scored_point.score, [scored_point]))
            return sorted(results, key=lambda x: x[0], reverse=True)[:limit]
        
        # For mixed results, continue with the original logic
        # Analyze query to determine if it's code-focused
        code_indicators = ['function', 'class', 'method', 'implement', 'code', 'bug', 
                            'error', 'syntax', 'variable', 'algorithm']
        issue_indicators = ['issue', 'pr', 'pull request', 'feature', 'enhancement', 
                            'proposal', 'discussion', 'roadmap']
        
        # Calculate query focus based on keyword presence
        code_focus = sum(1 for word in code_indicators if word in query_text.lower())
        issue_focus = sum(1 for word in issue_indicators if word in query_text.lower())
        
        # Adjust code ratio based on query analysis
        if code_focus > issue_focus:
            dynamic_code_ratio = min(0.9, code_ratio + 0.2)
        elif issue_focus > code_focus:
            dynamic_code_ratio = max(0.3, code_ratio - 0.2)
        else:
            dynamic_code_ratio = code_ratio
        
        # Separate code files and issues/PRs
        code_results = []
        issue_results = []
        
        for scored_point in search_result.points:
            score = scored_point.score
            
            # Create a tuple of (score, [scored_point]) to match expected output format
            result_tuple = (score, [scored_point])
            
            if scored_point.payload.get("type") == "code_file":
                code_results.append(result_tuple)
            else:
                issue_results.append(result_tuple)
        
        # If we found issues/PRs but the query seems code-focused, 
        # perform a secondary search to find related code files
        if issue_results and code_focus > issue_focus:
            # Extract key information from top issues
            issue_contexts = []
            for _, result_list in issue_results[:3]:
                result = result_list[0]  # Get the first result
                # Extract key phrases from issue content
                issue_content = result.payload.get("content", "")
                # Take first 300 chars for context summary
                issue_contexts.append(issue_content[:300])
            
            # Create expanded queries from issue contexts
            expanded_queries = [f"{query_text} {context}" for context in issue_contexts]
            
            # Perform searches with expanded queries
            for expanded_query in expanded_queries:
                sparse_emb, dense_emb = self.get_embeddings(expanded_query)
                
                # Search specifically for code files
                secondary_result = self.qdrant_client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        Prefetch(query=sparse_emb.as_object(), using="bm42", limit=5),
                        Prefetch(query=dense_emb.tolist(), using="jina", limit=5),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=5,
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="type",
                                match=models.MatchValue(value="code_file")
                            )
                        ]
                    )
                )
                
                # Add secondary results with slight penalty
                for scored_point in secondary_result.points:
                    score = scored_point.score * 0.85  # Penalty for secondary search
                    code_results.append((score, [scored_point]))
        
        # Remove duplicates from code_results while preserving highest score
        seen_ids = set()
        unique_code_results = []
        for score, result_list in sorted(code_results, key=lambda x: x[0], reverse=True):
            result = result_list[0]  # Get the first result
            result_id = result.id
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_code_results.append((score, result_list))
        
        # Calculate how many results of each type to include
        code_count = min(int(limit * dynamic_code_ratio), len(unique_code_results))
        issue_count = min(limit - code_count, len(issue_results))
        
        # Combine results according to the calculated ratio
        final_results = unique_code_results[:code_count] + issue_results[:issue_count]
        
        # Sort by score and return
        return sorted(final_results, key=lambda x: x[0], reverse=True)

    async def process_repository(self):
        """
        Process the entire repository, chunk it, and store in Qdrant.
        """
        delete_github_repo()
        
        # 1. Set up Qdrant collection
        self.setup_qdrant_collection()
        
        # 2. Fetch repository content
        await self.fetch_repository()
        
        # 3. Fetch issues and PRs
        issues_fetched = self.fetch_issues_and_prs()
        
        # 4. Parse code files
        code_chunks = self.parse_code_files()
        
        # 5. Parse issues and PRs if fetched successfully
        issue_pr_chunks = []
        if issues_fetched:
            issue_pr_chunks = self.parse_issues_and_prs()
        
        # 6. Combine all chunks
        all_chunks = code_chunks + issue_pr_chunks
        
        # 7. Store chunks in Qdrant
        self.store_chunks_in_qdrant(all_chunks)
        
        return {
            "code_files_count": len(code_chunks),
            "issues_prs_count": len(issue_pr_chunks),
            "total_chunks": len(all_chunks),
            "collection_name": self.collection_name
        }

def delete_github_repo(directory="."):
    """
    Deletes any directory named 'github-repo' in the given directory.
    If no directory is provided, it searches in the current working directory.
    """
    target_path = os.path.join(directory, "github-repo")

    if os.path.exists(target_path) and os.path.isdir(target_path):
        shutil.rmtree(target_path)
        print(f"Deleted: {target_path}")
    else:
        print(f"No 'github-repo' directory found in {directory}")

async def main():
    # Delete any existing 'github-repo' directory
    delete_github_repo()
    # Get user inputs
    repo_url = input("Enter GitHub repository URL (e.g., https://github.com/Hasan-Syed25/CPO_SIMPO): ")
    collection_name = input("Enter Qdrant collection name: ")
    
    # Create chunker and process repository
    chunker = GitHubRepoChunker(
        repo_url=repo_url,
        collection_name=collection_name
    )
    
    results = await chunker.process_repository()
    
    print("\nRepository Processing Summary:")
    print(f"Total code files processed: {results['code_files_count']}")
    print(f"Total issues/PRs processed: {results['issues_prs_count']}")
    print(f"Total chunks stored in Qdrant: {results['total_chunks']}")
    print(f"Qdrant collection name: {results['collection_name']}")
    
    # Demo search
    try_search = input("\nWould you like to try a search query? (y/n): ")
    if try_search.lower() == 'y':
        query = input("Enter your search query: ")
        search_results = chunker.search_repository(query, only_type="issues", limit=23)
    
        for i, ( _ , result ) in enumerate(search_results):
            result_type = result[0].payload.get("type", "unknown")
            if result_type == "code_file":
                filename = result[0].payload.get("metadata", {}).get("filename", "unknown")
                print(f"{i+1}. [{result_type}] {filename}\n\n{result[0].payload.get('content', '')[:500]}...\n")
            else:
                issue_type = result[0].payload.get("metadata", {}).get("issue_type", "unknown")
                issue_number = result[0].payload.get("metadata", {}).get("issue_number", "unknown")
                print(f"{i+1}. [{result_type}] {issue_type} #{issue_number}\n\n{result[0].payload.get('content', '')[:500]}...\n")
    
    print("\nProcess completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())