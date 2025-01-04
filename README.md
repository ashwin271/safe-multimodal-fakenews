# SAFE - AI-Powered Fake News Detector

An advanced Python tool leveraging Together AI's LLaMA Vision model and Tavily's fact-checking API to detect fake news through multi-modal analysis of text and images, cross-referencing claims against reliable sources.

## Background
This project was motivated by the paper "SAFE: Similarity-Aware Multi-Modal Fake News Detection" by Xinyi Zhou, Jindi Wu, and Reza Zafarani. The paper highlights the importance of analyzing the relationship between textual and visual information in news articles to effectively detect fake news. The proposed method in the paper, which focuses on identifying mismatches between text and images, influenced the development of this tool. You can read more about the paper [here](https://doi.org/10.48550/arXiv.2003.04981).

## Setup

### 1. API Key Configuration
1. Create accounts on [Together AI](https://api.together.ai) and [Tavily](https://tavily.com)
2. Generate API keys from both platforms
3. In the `src` directory, create or update the `.env` file:
   ```env
   TOGETHER_API_KEY=your_together_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

### 2. Environment Setup 

#### Option A: Direct Installation
```bash
pip install -r requirements.txt
```

#### Option B: Virtual Environment
```bash
python -m venv env
# Activate the environment (OS-specific)
pip install -r requirements.txt
```

## Usage

1. Add test images to the `data` directory
2. Execute the script:
```bash
cd src
python main.py
```

The tool will:
- Analyze image-text consistency
- Fact-check claims using reliable sources
- Assess the likelihood of fake news
- Provide detailed reasoning for the assessment

## Example Output
```json
{
  "image_text_match": "Yes",
  "image_text_match_confidence": 0.8,
  "fact_check": "Supported",
  "fact_check_confidence": 0.75,
  "fake_news": "No",
  "fake_news_confidence": 0.9,
  "reasoning": "The image analysis supports the news text with 80% confidence. Fact-checking results are supported with 75% confidence. Based on this analysis, the news is determined to be authentic with 90% confidence.",
  "fact_check_results": [
    {
      "title": "Fact Check: Trump's Relations with India",
      "url": "https://example.com/fact-check",
      "content": "Verified positive relations between Trump and Indian government.",
      "score": 0.85
    }
  ],
  "image_analysis": {
    "description": "Image shows Trump meeting with Indian officials.",
    "confidence": 0.8
  }
}
```

## Project Structure
```
.
├── LICENSE.md
├── README.md
├── requirements.txt
└── src
    ├── .env
    └── main.py
```

## Requirements
- Python 3.7+
- Dependencies listed in `requirements.txt`

## Features
- Multi-modal analysis (image and text)
- Integration with Tavily's advanced search API
- Focused search across reputable news and fact-checking sources
- Detailed assessment with confidence scores
- Asynchronous processing for improved performance

## How It Works
1. Processes input news claim and associated image
2. Utilizes Tavily API for fact-checking against reliable sources
3. Analyzes image content using LLaMA Vision model
4. Combines analyses to:
   - Evaluate image-text consistency
   - Cross-reference claims with fact-checking results
   - Generate a comprehensive assessment with reasoning

## Contributing
Contributions via issues or pull requests are welcome to enhance the project.

## License
This project is open source, available under the MIT License.
