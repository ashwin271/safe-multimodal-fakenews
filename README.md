# SAFE - Fake News Detection

A Python-based tool that uses Together AI's LLaMA Vision model to detect fake news by analyzing both text and images.

## Setup

### 1. API Key Setup
1. Visit [Together AI](https://api.together.ai) and create an account
2. Generate an API key from your dashboard
3. If a `.env` file does not exist in the `src` directory, create one:
   ```env
   TOGETHER_API_KEY=your_api_key_here
   ```
   If a dummy `.env` file already exists, update the `TOGETHER_API_KEY` with your actual API key.

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

The script will analyze the provided news text and image, and output:
- Whether the image matches the text
- Whether the news is likely fake
- Detailed reasoning for the assessment

## Example Output
```
Model's Response:
IMAGE-TEXT MATCH: No
FAKE NEWS: Yes
REASONING: The image shows plants or vegetation, which has no relation to the claim about teleportation...
```

## Project Structure
```bash
.
├── README.md       # Project documentation
├── data/           # Store your test images here
├── requirements.txt # Python dependencies
└── src/
    ├── .env        # API key configuration
    └── main.py     # Main detection script
```

## Requirements
- Python 3.7+
- python-dotenv
- together

