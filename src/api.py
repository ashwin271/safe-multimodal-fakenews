from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Depends
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
import json
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Together
from langchain_community.callbacks.manager import get_openai_callback
from tavily import TavilyClient
from dotenv import load_dotenv
import logging
import os
import base64
from typing import Optional, List
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pathlib
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fake_news_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Fake News Detection API")

# After creating the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize clients
tavily_client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))
together_model = Together(
    together_api_key=os.getenv('TOGETHER_API_KEY'),
    model="meta-llama/Llama-Vision-Free",  # or another appropriate model
    temperature=0.7,
)

# Mount static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

@app.get("/")
async def read_root():
    """Serve the index.html file"""
    return FileResponse('src/static/index.html')

class FactCheckResult(BaseModel):
    source: str
    relevance: float
    snippet: str

class FakeNewsAnalysis(BaseModel):
    image_text_match: bool
    fact_check_status: str  # "Supported" | "Contradicted" | "Inconclusive"
    is_fake_news: bool
    reasoning: str
    confidence_score: float
    fact_check_sources: List[FactCheckResult]

class NewsRequest(BaseModel):
    news_text: str

async def validate_image(image: UploadFile):
    """Validate the uploaded image"""
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Add size validation if needed
    contents = await image.read()
    if len(contents) > 5 * 1024 * 1024:  # 5MB limit
        raise HTTPException(status_code=400, detail="Image too large")
    
    await image.seek(0)  # Reset file pointer
    return contents

def fact_check_with_tavily(news_text: str) -> List[FactCheckResult]:
    """
    Search the internet for related facts about the news claim.
    """
    logger.info(f"Starting fact check for text: {news_text[:100]}...")
    try:
        search_result = tavily_client.search(
            query=news_text,
            search_depth="advanced",
            include_domains=["reuters.com", "apnews.com", "bbc.com", "factcheck.org", "snopes.com"],
            max_results=1
        )
        
        fact_check_results = [
            FactCheckResult(
                source=result.get('url', ''),
                relevance=result.get('relevance_score', 0.0),
                snippet=result.get('content', '')
            )
            for result in search_result.get('results', [])
        ]
        
        logger.info(f"Found {len(fact_check_results)} fact check results")
        return fact_check_results
    except Exception as e:
        logger.error(f"Tavily search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fact checking failed: {str(e)}")

# Create Langchain prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a fake news detection expert. Your task is to:
    1. Analyze the provided image and describe its contents
    2. Compare the image content with the news text
    3. Check if the facts align with the provided fact-checking results
    4. Determine if this is likely fake news

    Provide your analysis in this exact JSON format:
    {{
        "image_text_match": boolean,
        "fact_check_status": "Supported" | "Contradicted" | "Inconclusive",
        "is_fake_news": boolean,
        "reasoning": string,
        "confidence_score": float (0-1),
        "fact_check_sources": [{{"source": string, "relevance": float, "snippet": string}}]
    }}
    """),
    ("user", """News Text: {news_text}

    Image Analysis:
    Please analyze this image: {image_base64}
    
    Fact-Checking Results:
    {fact_check_results}
    
    Based on the image content, news text, and fact-checking results, please provide your analysis in the specified JSON format."""),
])

# Create output parser
output_parser = PydanticOutputParser(pydantic_object=FakeNewsAnalysis)

@app.post("/analyze", response_model=FakeNewsAnalysis)
async def analyze_news(
    news: str = Form(...), 
    image: UploadFile = File(...)
):
    logger.info("Starting news analysis request")
    
    try:
        # Validate image first
        image_contents = await validate_image(image)
        
        # Parse the news JSON string into NewsRequest
        news_data = json.loads(news)
        news_request = NewsRequest(**news_data)

        # Encode image
        image_base64 = base64.b64encode(image_contents).decode('utf-8')
        image_base64 = f"data:image/jpeg;base64,{image_base64}"
        
        # Get fact checking results
        fact_check_results = fact_check_with_tavily(news_request.news_text)
        
        # Log inputs to Together model
        logger.debug(f"Invoking Together model with inputs: news_text={news_request.news_text}, "
                     f"fact_check_results={fact_check_results}, image_base64={image_base64[:100]}...")

        # Create Langchain chain
        chain = prompt_template | together_model | output_parser
        
        # Track token usage
        with get_openai_callback() as cb:
            result = chain.invoke({
                "news_text": news_request.news_text,
                "fact_check_results": fact_check_results,
                "image_base64": image_base64
            })
            
            logger.info(f"Token usage - Total: {cb.total_tokens}, "
                       f"Prompt: {cb.prompt_tokens}, "
                       f"Completion: {cb.completion_tokens}")
        
        logger.info("Analysis completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)