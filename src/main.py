# src/main.py

import logging
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from typing import List
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import base64
from dotenv import load_dotenv
from models import FactCheckResult, FakeNewsAnalysis  # Import from models.py
from utils import (
    initialize_tavily_client,
    initialize_together_client,
    fact_check_with_tavily,
    analyze_with_together,
    validate_image_file
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("fake_news_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="SAFE - Fake News Detection API")

# Enable CORS (Adjust origins as needed for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize API clients
tavily_client = initialize_tavily_client()
together_client = initialize_together_client()

@app.get("/")
async def root():
    """Serve the index.html for the homepage."""
    logger.info("Serving homepage.")
    return FileResponse("static/index.html")

@app.post("/analyze", response_model=FakeNewsAnalysis)
async def analyze_news(
    news: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Analyze the provided news text and image for fake news.
    """
    logger.info("Received analysis request.")

    try:
        # Parse and validate news text
        news_data = {"news_text": news}
        news_text = news_data["news_text"]
        logger.info(f"News text received: {news_text[:100]}...")

        # Validate and process image
        image_contents = await validate_image_file(image)
        image_base64 = base64.b64encode(image_contents).decode("utf-8")
        image_data_uri = f"data:{image.content_type};base64,{image_base64}"
        logger.info(f"Image {image.filename} processed and encoded.")

        # Perform fact-checking with Tavily
        fact_check_results = fact_check_with_tavily(tavily_client, news_text)
        logger.info(f"Fact-checking completed with {len(fact_check_results)} results.")

        # Analyze image and text with Together AI
        together_response = analyze_with_together(
            together_client,
            news_text,
            image_data_uri
        )
        logger.info("Image-text analysis with Together AI completed.")

        # Combine results
        image_text_match = together_response.get("image_text_match", False)
        fact_check_status = together_response.get("fact_check_status", "Inconclusive")
        is_fake_news = together_response.get("is_fake_news", True)
        reasoning = together_response.get("reasoning", "Insufficient data to determine authenticity.")
        confidence_score = together_response.get("confidence_score", 0.0)

        # Override is_fake_news based on fact-check status if necessary
        if fact_check_status == "Contradicted":
            is_fake_news = True
        elif fact_check_status == "Supported":
            is_fake_news = False
        # If "Inconclusive", retain the value from Together's analysis

        # **HERE IS THE FIX: Convert FactCheckResult instances to dictionaries**
        fact_check_sources_dict = [result.dict() for result in fact_check_results]

        analysis_result = FakeNewsAnalysis(
            image_text_match=image_text_match,
            fact_check_status=fact_check_status,
            is_fake_news=is_fake_news,
            reasoning=reasoning,
            confidence_score=confidence_score,
            fact_check_sources=fact_check_sources_dict,  # Pass as dicts
        )

        logger.info("Analysis successful.")
        return analysis_result

    except HTTPException as http_err:
        logger.error(f"HTTP error during analysis: {http_err.detail}")
        raise http_err
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during analysis.")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.info("Health check requested.")
    return {"status": "healthy"}

# Start the server if this file is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
