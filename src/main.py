from together import Together
from dotenv import load_dotenv
import os
import base64

# Load environment variables from .env file
load_dotenv()

# Initialize the Together API client
client = Together(api_key=os.getenv('TOGETHER_API_KEY'))

def detect_fake_news(news_text, image_path):
    """
    Function to detect fake news using LLAMA 3.2 hosted on the Together API.

    Args:
        news_text (str): The text of the news article.
        image_path (str): Path to the image associated with the news.

    Returns:
        str: Model's response indicating if the news is fake or true.
    """
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
            "content": """You are a fake news detection expert. Analyze the provided news text and image, then respond in the following format:

IMAGE-TEXT MATCH: [Yes/No]
FAKE NEWS: [Yes/No]
REASONING: [Your detailed explanation]

Be direct and concise in your assessment."""
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"News Text: {news_text}\nPlease analyze if this news is authentic by checking if the image supports the text."},
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
