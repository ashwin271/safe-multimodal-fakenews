import os
import base64
import json
import asyncio
import logging
from dotenv import load_dotenv
from tavily import (
    TavilyClient,
    MissingAPIKeyError,
    InvalidAPIKeyError,
    UsageLimitExceededError
)
from together import Together
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Load environment variables from .env file
load_dotenv()

# Initialize Together AI client
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
if not TOGETHER_API_KEY:
    logging.error("Together API key is missing. Please set it in the .env file.")
    raise EnvironmentError("Together API key is missing.")

together = Together(api_key=TOGETHER_API_KEY)

# Initialize Tavily client with error handling
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
if not TAVILY_API_KEY:
    logging.error("Tavily API key is missing. Please set it in the .env file.")
    raise EnvironmentError("Tavily API key is missing.")

try:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
except InvalidAPIKeyError:
    logging.error("Invalid Tavily API key. Please verify your key.")
    raise EnvironmentError("Invalid Tavily API key.")
except MissingAPIKeyError:
    logging.error("Missing Tavily API key.")
    raise EnvironmentError("Missing Tavily API key.")
except Exception as e:
    logging.error(f"Unexpected error initializing TavilyClient: {e}")
    raise RuntimeError(f"Unexpected error initializing TavilyClient: {e}")

# Define Pydantic models for JSON Output
class FactCheckResult(BaseModel):
    title: str
    url: str
    content: str
    score: float

class ImageAnalysisResult(BaseModel):
    description: str
    confidence: float

class FakeNewsAssessment(BaseModel):
    image_text_match: str  # "Yes" or "No"
    image_text_match_confidence: float  # 0.0 to 1.0
    fact_check: str  # "Supported", "Contradicted", "Inconclusive"
    fact_check_confidence: float  # 0.0 to 1.0
    fake_news: str  # "Yes" or "No"
    fake_news_confidence: float  # 0.0 to 1.0
    reasoning: str
    fact_check_results: List[FactCheckResult]
    image_analysis: ImageAnalysisResult

# Create a ThreadPoolExecutor for synchronous I/O operations
executor = ThreadPoolExecutor(max_workers=5)

async def run_in_executor(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)

# Asynchronous Fact-Checking Function
async def fact_check(news_text: str, max_results: int = 5) -> Optional[dict]:
    """
    Asynchronously search the internet for related facts about the news claim using Tavily's API.
    """
    try:
        logging.info("Starting fact-checking...")
        search_result = await run_in_executor(
            lambda: tavily_client.search(
                query=news_text,
                search_depth="advanced",
                include_domains=["reuters.com", "apnews.com", "bbc.com", "factcheck.org", "snopes.com"],
                max_results=max_results
            )    
        )
        logging.info("Fact-checking completed.")
        return search_result
    except UsageLimitExceededError:
        logging.error("Tavily usage limit exceeded.")
    except InvalidAPIKeyError:
        logging.error("Invalid Tavily API key.")
    except MissingAPIKeyError:
        logging.error("Missing Tavily API key.")
    except Exception as e:
        logging.error(f"Tavily search error: {e}")
    return None

# Encode Image to Base64
def encode_image(image_path: str) -> Optional[str]:
    """
    Encode the image at the given path to a base64 string.
    """
    if not os.path.exists(image_path):
        logging.error(f"Image file not found: {image_path}")
        return None
    try:
        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            return f"data:image/jpeg;base64,{image_base64}"
    except Exception as e:
        logging.error(f"Error encoding image: {e}")
        return None

# Asynchronous Image Analysis Function
async def analyze_image(image_base64: str) -> Optional[str]:
    """
    Analyze the image using Together AI's Vision Model.
    """
    messages = [
        {
            "role": "system",
            "content": "You are an image analysis expert. Describe the contents of the following image."
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_base64}}
            ]
        }
    ]

    try:
        logging.info("Starting image analysis...")
        response = await run_in_executor(
            lambda: together.chat.completions.create(
                model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                messages=messages,
                max_tokens=256,
                temperature=0.5,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.2,
                stop=["<|end|>"],
                stream=False
            )
        ) 
        image_description = response.choices[0].message.content.strip()
        logging.info(f"Image analysis completed: {image_description}")
        return image_description
    except Exception as e:
        logging.error(f"Error communicating with Together AI Vision Model: {e}")
        return None

# Function to assess image-text match
def assess_image_text_match(image_description: str, news_text: str) -> tuple[str, float]:
    """
    Determines if the image supports the news text.
    """
    # For demonstration, implement a simple keyword-based matching.
    # This can be enhanced with more sophisticated NLP techniques.
    match = False
    keywords = ["supports", "aligns", "corroborates", "indicates", "relates"]
    
    # Check if any keyword is present in both the image description and news text
    for keyword in keywords:
        if keyword in image_description.lower() and keyword in news_text.lower():
            match = True
            break
    
    # Assign confidence based on match presence
    if match:
        return "Yes", 0.8  # Example confidence score
    else:
        return "No", 0.6  # Example confidence score

# Enhanced Fake News Detection Function
async def detect_fake_news(news_text: str, image_path: str) -> Optional[FakeNewsAssessment]:
    """
    Enhanced fake news detection with Tavily fact-checking and image correlation.
    
    Args:
        news_text (str): The text of the news article.
        image_path (str): Path to the image associated with the news.
    
    Returns:
        FakeNewsAssessment: Structured JSON output with detailed assessment.
    """
    # Perform fact-checking asynchronously
    fact_check_results = await fact_check(news_text)
    if not fact_check_results:
        logging.error("Fact-checking failed or returned no results.")
        return None

    # Parse fact-check results into Pydantic models
    parsed_fact_checks = []
    for result in fact_check_results.get('results', []):
        try:
            parsed_fact = FactCheckResult(
                title=result.get('title', ''),
                url=result.get('url', ''),
                content=result.get('content', ''),
                score=result.get('score', 0.0)
            )
            parsed_fact_checks.append(parsed_fact)
        except ValidationError as ve:
            logging.warning("Validation Error in FactCheckResult:", ve)
            continue

    if not parsed_fact_checks:
        logging.error("No valid fact-check results available.")
        return None

    # Encode the image
    image_base64 = encode_image(image_path)
    if not image_base64:
        logging.error("Image encoding failed.")
        return None

    # Analyze the image
    image_description = await analyze_image(image_base64)
    if not image_description:
        logging.error("Image analysis failed.")
        return None

    # Assess image-text match
    image_text_match, image_text_match_confidence = assess_image_text_match(image_description, news_text)

    # Determine fact check summary
    # Calculate average score from fact-check results
    average_score = sum(fc.score for fc in parsed_fact_checks) / len(parsed_fact_checks)
    
    if average_score >= 0.7:
        fact_check_status = "Supported"
        fact_check_confidence = average_score
    elif average_score <= 0.3:
        fact_check_status = "Contradicted"
        fact_check_confidence = 1.0 - average_score  # Higher confidence if lower average score
    else:
        fact_check_status = "Inconclusive"
        fact_check_confidence = 0.5  # Moderate confidence

    # Determine fake news based on fact check and image-text match
    if fact_check_status == "Contradicted":
        fake_news = "Yes"
        fake_news_confidence = 0.9
    elif fact_check_status == "Supported":
        fake_news = "No"
        fake_news_confidence = 0.9
    else:
        fake_news = "Inconclusive"
        fake_news_confidence = 0.6

    # Compile reasoning
    reasoning = (
        f"The image analysis indicates that the image {'supports' if image_text_match == 'Yes' else 'does not support'} the news text with a confidence of {image_text_match_confidence*100:.1f}%. "
        f"Fact-checking results are {fact_check_status.lower()} with an average confidence of {fact_check_confidence*100:.1f}%. "
        f"Based on this analysis, the news is determined to be {'fake' if fake_news == 'Yes' else 'authentic' if fake_news == 'No' else 'inconclusive'} with a confidence of {fake_news_confidence*100:.1f}%."
    )

    # Compile assessment
    assessment = FakeNewsAssessment(
        image_text_match=image_text_match,
        image_text_match_confidence=image_text_match_confidence,
        fact_check=fact_check_status,
        fact_check_confidence=fact_check_confidence,
        fake_news=fake_news,
        fake_news_confidence=fake_news_confidence,
        reasoning=reasoning,
        fact_check_results=parsed_fact_checks,
        image_analysis=ImageAnalysisResult(
            description=image_description,
            confidence=image_text_match_confidence
        )
    )

    return assessment

# Main Execution Function
async def main():
    # Example usage
    image_name = "trump.jpg"
    news_text = "Trump has good relations with the Indian Government."

    # Get the absolute path to the data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    image_path = os.path.join(project_root, "data", image_name)

    # Detect fake news
    assessment = await detect_fake_news(news_text, image_path)
    if assessment:
        # Convert Pydantic model to JSON
        assessment_json = assessment.model_dump_json(indent=2)
        logging.info("Fake News Assessment:")
        print(assessment_json)
    else:
        logging.error("Failed to assess the fake news.")

# Entry Point
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")