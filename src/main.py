from together import Together
from dotenv import load_dotenv
from tavily import TavilyClient
import os
import base64

# Load environment variables from .env file
load_dotenv()

# Initialize the Together API client
client = Together(api_key=os.getenv('TOGETHER_API_KEY'))

# Initialize Tavily client
tavily_client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))

def fact_check_with_tavily(news_text):
    """
    Search the internet for related facts about the news claim.
    """
    try:
        # Search with Tavily's news-focused search
        search_result = tavily_client.search(
            query=news_text,
            search_depth="advanced",
            include_domains=["reuters.com", "apnews.com", "bbc.com", "factcheck.org", "snopes.com"],
            max_results=5
        )
        
        return search_result
    except Exception as e:
        print(f"Tavily search error: {e}")
        return None

def detect_fake_news(news_text, image_path):
    """
    Enhanced fake news detection with Tavily fact-checking.

    Args:
        news_text (str): The text of the news article.
        image_path (str): Path to the image associated with the news.

    Returns:
        str: Model's response indicating if the news is fake or true.
    """
    # First, get fact-checking results
    fact_check_results = fact_check_with_tavily(news_text)

    # Validate image path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load the image file
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        image_base64 = f"data:image/jpeg;base64,{image_base64}"

    # Prepare the input for the model
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
                {"type": "text", "text": f"""News Text: {news_text}
                
Fact-Checking Results:
{fact_check_results}

Please analyze if this news is authentic by checking:
1. If the image supports the text
2. If the fact-checking results support or contradict the claim"""},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_base64
                    }
                }
            ]
        }
    ]

    # Call the Together API
    response = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        messages=messages,
        max_tokens=512,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1.0,
        stop=["<|eot_id|>", "<|eom_id|>"],
        stream=False
    )

    return response.choices[0].message.content

if __name__ == "__main__":

    # Example usage
    image_name = "plants.jpg"
    news_text = "A new scientific breakthrough claims that teleportation is possible."

    # Get the absolute path to the data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    image_path = os.path.join(project_root, "data", image_name)

    # Detect fake news
    result = detect_fake_news(news_text, image_path)
    print(f"Model's Response:\n{result}")
