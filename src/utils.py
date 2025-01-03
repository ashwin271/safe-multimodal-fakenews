# src/utils.py

import os
import logging
import base64
from fastapi import HTTPException, UploadFile
from typing import List

from tavily import TavilyClient
from together import Together

from models import FactCheckResult  # Import from models.py

# Configure logger for utilities
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def initialize_tavily_client() -> TavilyClient:
    """Initialize and return the Tavily client."""
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        logger.error("TAVILY_API_KEY not found in environment variables.")
        raise EnvironmentError("TAVILY_API_KEY not set.")
    return TavilyClient(api_key=tavily_api_key)

def initialize_together_client() -> Together:
    """Initialize and return the Together AI client."""
    together_api_key = os.getenv("TOGETHER_API_KEY")
    if not together_api_key:
        logger.error("TOGETHER_API_KEY not found in environment variables.")
        raise EnvironmentError("TOGETHER_API_KEY not set.")
    return Together(api_key=together_api_key)

async def validate_image_file(image: UploadFile) -> bytes:
    """Validate the uploaded image file."""
    if not image.content_type.startswith("image/"):
        logger.warning("Uploaded file is not an image.")
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    contents = await image.read()
    if len(contents) > 5 * 1024 * 1024:  # 5MB limit
        logger.warning("Uploaded image exceeds size limit.")
        raise HTTPException(status_code=400, detail="Image file size exceeds 5MB.")
    
    await image.seek(0)  # Reset file pointer for further use
    return contents

def fact_check_with_tavily(client: TavilyClient, news_text: str) -> List[FactCheckResult]:
    """Retrieve fact-checking results for the given news text."""
    try:
        logger.info(f"Initiating fact-checking for news: {news_text[:100]}...")
        search_result = client.search(
            query=news_text,
            search_depth="advanced",
            include_domains=[
                "reuters.com",
                "apnews.com",
                "bbc.com",
                "factcheck.org",
                "snopes.com"
            ],
            max_results=5
        )
        results = search_result.get("results", [])
        fact_checks = [
            FactCheckResult(
                source=result.get("url", "Unknown Source"),
                relevance=result.get("relevance_score", 0.0),
                snippet=result.get("content", "No snippet available.")
            )
            for result in results
        ]
        logger.info(f"Retrieved {len(fact_checks)} fact-checking results.")
        return fact_checks
    except Exception as e:
        logger.error(f"Tavily fact-checking error: {str(e)}")
        raise HTTPException(status_code=500, detail="Fact-checking service failed.")

def analyze_with_together(client: Together, news_text: str, image_data_uri: str) -> dict:
    """
    Analyze the news text and image using Together AI's LLaMA Vision model.

    Args:
        client (Together): Initialized Together AI client.
        news_text (str): The news text to analyze.
        image_data_uri (str): Base64 encoded image data URI.

    Returns:
        dict: Parsed response from Together AI's model.
    """
    try:
        logger.info("Preparing messages for Together AI analysis.")
        messages = [
            {
                "role": "system",
                "content": """You are a fake news detection expert. Analyze the provided news text, image, and fact-checking results. Respond in the following format:

IMAGE-TEXT MATCH: [Yes/No]
FACT CHECK: [Supported/Contradicted/Inconclusive]
FAKE NEWS: [Yes/No]
REASONING: [Your detailed explanation including references to fact-checking results]

Be direct and concise in your assessment."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"News Text: {news_text}"},
                    {"type": "image_url", "image_url": {"url": image_data_uri}}
                ]
            }
        ]

        logger.info("Sending request to Together AI's LLaMA Vision model.")
        response = client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.0,
            stop=["<|eot_id|>", "<|eom_id|>"],
            stream=False
        )

        # Parse the response content
        analysis_content = response.choices[0].message.content.strip()
        logger.info("Received response from Together AI.")

        # Example response parsing (Adjust based on actual response format)
        analysis_lines = analysis_content.split("\n")
        analysis_dict = {}
        for line in analysis_lines:
            if ":" in line:
                key, value = line.split(":", 1)
                analysis_dict[key.strip().lower().replace(" ", "_")] = value.strip()

        # Convert relevant fields
        image_text_match = analysis_dict.get("image-text_match", "No").lower() == "yes"
        fact_check_status = analysis_dict.get("fact_check", "Inconclusive")
        is_fake_news = analysis_dict.get("fake_news", "Yes").lower() == "yes"
        reasoning = analysis_dict.get("reasoning", "")
        confidence_score = analysis_dict.get("confidence_score", 0.0) or 0.0  # Ensure float

        return {
            "image_text_match": image_text_match,
            "fact_check_status": fact_check_status,
            "is_fake_news": is_fake_news,
            "reasoning": reasoning,
            "confidence_score": float(confidence_score)  # Ensure float type
        }

    except Exception as e:
        logger.error(f"Together AI analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail="Image-text analysis failed.")
