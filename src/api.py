import logging
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
import json
from typing import Optional, List
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import base64
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("fake_news_detector.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Fake News Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")


# Define data models
class FactCheckResult(BaseModel):
    source: str
    relevance: float
    snippet: str


class FakeNewsAnalysis(BaseModel):
    image_text_match: bool
    fact_check_status: str
    is_fake_news: bool
    reasoning: str
    confidence_score: float
    fact_check_sources: List[FactCheckResult]


class NewsRequest(BaseModel):
    news_text: str


# Tavily client setup
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@app.get("/")
async def root():
    """Serve index.html for the homepage."""
    return FileResponse("src/static/index.html")


async def validate_image(image: UploadFile):
    """Validate and ensure the uploaded file is an image."""
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    contents = await image.read()
    if len(contents) > 5 * 1024 * 1024:  # 5MB limit
        raise HTTPException(status_code=400, detail="Image file size exceeds 5MB")
    await image.seek(0)  # Reset file pointer for further use
    return contents


def fact_check_with_tavily(news_text: str) -> List[FactCheckResult]:
    """Retrieve fact-checking results for the given news text."""
    try:
        logger.info(f"Fact-checking news: {news_text[:100]}...")
        search_result = tavily_client.search(
            query=news_text,
            search_depth="advanced",
            include_domains=["reuters.com", "apnews.com", "bbc.com", "factcheck.org", "snopes.com"],
            max_results=3,
        )
        return [
            FactCheckResult(
                source=result.get("url", ""),
                relevance=result.get("relevance_score", 0.0),
                snippet=result.get("content", ""),
            )
            for result in search_result.get("results", [])
        ]
    except Exception as e:
        logger.error(f"Tavily fact-check error: {e}")
        raise HTTPException(status_code=500, detail="Fact-checking failed")


@app.post("/analyze", response_model=FakeNewsAnalysis)
async def analyze_news(news: str = Form(...), image: UploadFile = File(...)):
    """Analyze the provided news text and image for fake news."""
    try:
        # Validate and process image
        image_contents = await validate_image(image)
        image_base64 = base64.b64encode(image_contents).decode("utf-8")
        image_base64 = f"data:image/jpeg;base64,{image_base64}"

        # Parse news text
        news_request = NewsRequest(**json.loads(news))

        # Perform fact-checking
        fact_check_results = fact_check_with_tavily(news_request.news_text)

        # Analyze fact-check results
        total_relevance = sum(result.relevance for result in fact_check_results)
        average_relevance = total_relevance / len(fact_check_results) if fact_check_results else 0.0

        # Determine fact-check status
        if average_relevance > 0.7:
            fact_check_status = "Supported"
            is_fake_news = False
        elif 0.3 <= average_relevance <= 0.7:
            fact_check_status = "Inconclusive"
            is_fake_news = True  # Treat as potential fake news due to lack of strong evidence
        else:
            fact_check_status = "Contradicted"
            is_fake_news = True

        # Calculate confidence score
        confidence_score = round(average_relevance * 100, 2)

        # Generate reasoning
        reasoning = (
            "The news is marked as 'Supported' based on high relevance of fact-check sources."
            if fact_check_status == "Supported"
            else "The analysis indicates inconclusive or contradicting evidence based on fact-check results."
        )

        # Return analysis result
        analysis_result = FakeNewsAnalysis(
            image_text_match=True,
            fact_check_status=fact_check_status,
            is_fake_news=is_fake_news,
            reasoning=reasoning,
            confidence_score=confidence_score,
            fact_check_sources=fact_check_results,
        )

        logger.info("Analysis complete.")
        return analysis_result

    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# Start the server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
