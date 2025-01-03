# SAFE - Fake News Detection

A Python-based tool that uses Together AI's LLaMA Vision model and Tavily's fact-checking API to detect fake news by analyzing text, images, and verifying claims against reliable sources.

## Setup

### 1. API Key Setup
1. Visit [Together AI](https://api.together.ai) and create an account
2. Visit [Tavily](https://tavily.com) and create an account
3. Generate API keys from both dashboards
4. If a `.env` file does not exist in the `src` directory, create one:
   ```env
   TOGETHER_API_KEY=your_together_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```
   If a dummy `.env` file already exists, update both API keys with your actual values.

### 2. Environment Setup 

#### Option A: Direct Installation
Install required packages:
```bash
pip install -r requirements.txt
```

#### Option B: Virtual Environment
```bash
# Create virtual environment
python -m venv env

# Activate virtual environment
# On Windows:
env\Scripts\activate
# On macOS/Linux:
source env/bin/activate

# Install required packages
pip install -r requirements.txt
```

## Usage

1. Place your test images in the `data` directory
2. Run the script:
```bash
cd src
python main.py
```

The script will perform a comprehensive analysis of the news content:
- Verify if the image matches the text content
- Fact-check claims against reliable news sources (Reuters, AP News, BBC, FactCheck.org, Snopes)
- Determine if the news is likely fake
- Provide detailed reasoning for the assessment

## Example Output
```
Model's Response:
IMAGE-TEXT MATCH: No
FACT CHECK: Contradicted
FAKE NEWS: Yes
REASONING: The image shows plants or vegetation, which has no relation to the claim about teleportation. 
Additionally, fact-checking results from reliable sources show no evidence supporting this claim...
```

## Project Structure
```bash
.
├── README.md       # Project documentation
├── data/           # Store your test images here
├── requirements.txt # Python dependencies
└── src/
    ├── .env        # API keys configuration
    └── main.py     # Main detection script
```

## Requirements
- Python 3.7+
- python-dotenv
- together
- tavily-python

## Features
- Multi-modal analysis combining image and text verification
- Integration with Tavily's advanced search API for fact-checking
- Focused search across reliable news sources and fact-checking websites
- Detailed reasoning for each assessment
- Easy-to-understand output format

## How It Works
1. The system takes a news claim and associated image as input
2. Tavily API searches reliable sources for fact-checking information
3. LLaMA Vision model analyzes both the image and text content
4. The system combines these analyses to:
   - Verify image-text consistency
   - Cross-reference claims with fact-checking results
   - Provide a final assessment with detailed reasoning

## Contributing
Feel free to open issues or submit pull requests to improve the project.

## License
This project is open source and available under the MIT License.