from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import asyncio
import os
from main import detect_fake_news

app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse('static/index.html')

@app.get("/factcheck.html", response_class=HTMLResponse)
async def read_root():
    return FileResponse('static/factcheck.html')

@app.post("/detect_fake_news")
async def detect_fake_news_api(
    news_text: str = Form(...), 
    image: UploadFile = File(...)
):
    # Create a temporary directory if it doesn't exist
    os.makedirs("temp", exist_ok=True)
    
    # Save the uploaded image to a temporary location
    image_path = os.path.join("temp", f"temp_{image.filename}")
    with open(image_path, "wb") as image_file:
        content = await image.read()
        image_file.write(content)

    try:
        # Call the detect_fake_news function from main.py
        assessment = await detect_fake_news(news_text, image_path)
        
        if not assessment:
            return {"error": "Failed to assess the fake news."}
        
        # Return JSON response
        return assessment.model_dump()
    
    finally:
        # Clean up the temporary image file
        if os.path.exists(image_path):
            os.remove(image_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
