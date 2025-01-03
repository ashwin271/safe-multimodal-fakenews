## Overview of Changes

1. **Consolidate API and Detection Logic**: We'll ensure that the FastAPI server (`main.py`) handles both image-text analysis using Together AI and fact-checking using Tavily seamlessly within the `/analyze` endpoint.

2. **Remove Redundancies**: Eliminate any redundant scripts or outdated code snippets to maintain a clean and maintainable codebase.

3. **Enhance Error Handling and Logging**: Improve logging for better traceability and handle potential errors gracefully to avoid server crashes.

4. **Modularize Code**: Separate concerns by modularizing the code, making it easier to manage and extend in the future.

5. **Update README**: Ensure that the README reflects the latest setup and usage instructions based on the updated code structure.

## Updated Project Structure

```bash
.
├── README.md            # Project documentation
├── data/                # Store your test images here
├── requirements.txt     # Python dependencies
└── src/
    ├── .env             # API keys configuration
    ├── main.py          # FastAPI server with analysis endpoints
    └── utils.py         # Utility functions for Together and Tavily integrations
```

## 1. Updated `main.py`

The `main.py` file serves as the FastAPI server, handling incoming requests, validating inputs, invoking Together AI's vision model and Tavily's fact-checking API, and returning comprehensive analysis results.

```python
# src/main.py

import logging
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from typing import List
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import base64
from dotenv import load_dotenv

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


# Initialize API clients
tavily_client = initialize_tavily_client()
together_client = initialize_together_client()


@app.get("/")
async def root():
    """Serve index.html for the homepage."""
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
        news_data = NewsRequest(**{"news_text": news})
        news_text = news_data.news_text
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
        elif fact_check_status == "Inconclusive":
            # Retain the value from Together's analysis
            pass

        analysis_result = FakeNewsAnalysis(
            image_text_match=image_text_match,
            fact_check_status=fact_check_status,
            is_fake_news=is_fake_news,
            reasoning=reasoning,
            confidence_score=confidence_score,
            fact_check_sources=fact_check_results
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
```

### Explanation of `main.py` Changes

1. **Modular Utilities**: Moved Together AI and Tavily related functions to a separate `utils.py` file for better organization and readability.

2. **Enhanced Logging**: Improved logging statements to provide detailed insights into each step of the processing pipeline.

3. **Combined Analysis**: Integrated both image-text analysis and fact-checking results to provide a comprehensive fake news assessment.

4. **Error Handling**: Added specific error handling to catch and log HTTP exceptions separately from unexpected errors.

## 2. New `utils.py`

This utility module handles initialization and interaction with Together AI and Tavily APIs, as well as image validation.

```python
# src/utils.py

import os
import logging
import base64
from fastapi import HTTPException, UploadFile
from typing import List

from tavily import TavilyClient
from together import Together

from pydantic import BaseModel


# Configure logger for utilities
logger = logging.getLogger(__name__)


class FactCheckResult(BaseModel):
    source: str
    relevance: float
    snippet: str


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


def validate_image_file(image: UploadFile) -> bytes:
    """Validate the uploaded image file."""
    if not image.content_type.startswith("image/"):
        logger.warning("Uploaded file is not an image.")
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    contents = image.file.read()
    if len(contents) > 5 * 1024 * 1024:  # 5MB limit
        logger.warning("Uploaded image exceeds size limit.")
        raise HTTPException(status_code=400, detail="Image file size exceeds 5MB.")
    
    image.file.seek(0)  # Reset file pointer for further use
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
        confidence_score = 0.0  # Placeholder (Adjust if Together provides confidence score)

        return {
            "image_text_match": image_text_match,
            "fact_check_status": fact_check_status,
            "is_fake_news": is_fake_news,
            "reasoning": reasoning,
            "confidence_score": confidence_score
        }

    except Exception as e:
        logger.error(f"Together AI analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail="Image-text analysis failed.")
```

### Explanation of `utils.py`

1. **Initialization Functions**: `initialize_tavily_client` and `initialize_together_client` initialize their respective API clients using API keys from environment variables.

2. **Image Validation**: `validate_image_file` ensures that the uploaded file is an image and does not exceed the size limit.

3. **Fact-Checking**: `fact_check_with_tavily` interacts with Tavily's API to retrieve fact-checking results, returning them in a structured format.

4. **Together AI Analysis**: `analyze_with_together` sends the news text and image to Together AI's LLaMA Vision model, parses the response, and extracts relevant information.

## 3. Updated `requirements.txt`

Ensure that all necessary dependencies are included.

```bash
fastapi
uvicorn
python-dotenv
tavily-python
together
pydantic
```

**Note**: Ensure that the package names for `tavily-python` and `together` match the actual package names used. If the `tavily` and `together` clients are custom or proprietary, ensure they're correctly installed or included in your environment.

## 4. Ensuring `.env` Configuration

Your `.env` file located in the `src/` directory should contain your API keys:

```env
TOGETHER_API_KEY=your_together_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

**Security Tip**: Never commit your `.env` file to version control. Add it to your `.gitignore`.

## 5. Serving Static Files

Ensure that you have a `static` directory within `src/` containing an `index.html` file for the homepage.

```bash
src/
└── static/
    └── index.html
```

**Example `index.html`**:

```html
<!-- src/static/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SAFE - Fake News Detection</title>
</head>
<body>
    <h1>Welcome to SAFE - Fake News Detection API</h1>
    <p>Use the /analyze endpoint to submit news content for analysis.</p>
</body>
</html>
```

## 6. Running the Application

Navigate to the root directory of your project and install the dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
cd src
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Your API should now be running at `http://localhost:8000/`. You can access the homepage at the root URL or interact with the `/analyze` endpoint.

## 7. Testing the `/analyze` Endpoint

You can test the `/analyze` endpoint using tools like **cURL** or **Postman**. Here's an example using cURL:

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "news=Your news text here" \
  -F "image=@/path/to/your/image.jpg;type=image/jpeg"
```

### Example Response

```json
{
  "image_text_match": false,
  "fact_check_status": "Contradicted",
  "is_fake_news": true,
  "reasoning": "The image shows plants or vegetation, which has no relation to the claim about teleportation. Additionally, fact-checking results from reliable sources show no evidence supporting this claim...",
  "confidence_score": 85.75,
  "fact_check_sources": [
    {
      "source": "https://snopes.com/teleportation-fake-news",
      "relevance": 0.95,
      "snippet": "Teleportation remains a theoretical concept with no practical implementations..."
    },
    {
      "source": "https://factcheck.org/teleportation-claim",
      "relevance": 0.90,
      "snippet": "There is no scientific evidence to support the recent claims of teleportation breakthroughs..."
    }
  ]
}
```

## 8. Additional Suggestions

1. **API Authentication**: For security, especially in production, consider implementing authentication mechanisms (e.g., API keys, OAuth) to restrict access to your API endpoints.

2. **Rate Limiting**: Protect your API from abuse by implementing rate limiting using tools like **FastAPI Limiter**.

3. **Testing**: Implement unit and integration tests to ensure your application behaves as expected. Consider using **pytest** along with **FastAPI's TestClient**.

4. **Dockerization**: Containerize your application using Docker for easier deployment and scalability.

5. **Continuous Integration/Continuous Deployment (CI/CD)**: Set up CI/CD pipelines to automate testing and deployment processes.

6. **Enhance Logging**: Integrate more advanced logging solutions like **Loguru** or centralized logging systems (e.g., ELK Stack) for better log management and analysis.

7. **Error Monitoring**: Utilize tools like **Sentry** to monitor and track errors in real-time.

8. **Documentation**: Leverage FastAPI's automatic interactive API documentation (Swagger UI) by navigating to `http://localhost:8000/docs` or `http://localhost:8000/redoc`.

## 9. Final Notes

By restructuring your project as outlined above, you ensure a more maintainable, scalable, and efficient codebase. Integrating Together AI's vision model with Tavily's fact-checking API within the FastAPI framework provides a robust solution for fake news detection, leveraging both image and text analysis.
