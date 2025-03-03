import os
import openai
from dotenv import load_dotenv
load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_NAME")

llm_1 = openai.AsyncAzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    azure_deployment=AZURE_DEPLOYMENT,
    api_key=AZURE_API_KEY,
    api_version="2024-08-01-preview"
)

llm_2 = openai.AsyncAzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    azure_deployment=AZURE_DEPLOYMENT,
    api_key=AZURE_API_KEY,
    api_version="2024-08-01-preview"
)