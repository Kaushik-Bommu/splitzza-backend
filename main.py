import os
import cv2
import numpy as np
import easyocr
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

# Import the modern Google GenAI SDK
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 1. Load local environment variables from the .env file
load_dotenv()

# 2. Initialize EasyOCR once when the server boots up
print("🤖 Loading EasyOCR Models into memory...")
# Set gpu=True if your laptop has a dedicated NVIDIA graphics card
reader = easyocr.Reader(['en'], gpu=False) 

# 3. Initialize the Gemini Client
# It automatically reads the GEMINI_API_KEY variable loaded by dotenv
client = genai.Client()

# 4. Setup FastAPI Application
app = FastAPI(title="Splitzza AI Engine", version="1.0")

# Enable Cross-Origin Resource Sharing (CORS)
# This allows your local React app (port 3000) to securely talk to this API (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your exact frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. Define Pydantic Schemas for Strict JSON enforcement
class FoodItem(BaseModel):
    name: str = Field(description="Cleaned up name of the food item")
    quantity: float = Field(description="Quantity ordered")
    price: float = Field(description="Price per single unit")
    total_value: float = Field(description="Total value for this item")

class SplitzzaReceipt(BaseModel):
    restaurant_name: str
    items: List[FoodItem] = Field(description="Extract food/drink items. Ignore totals or headers.")
    subtotal: Optional[float] = Field(description="Subtotal before taxes")
    tax: Optional[float] = Field(description="Total tax amount combined")
    total: float = Field(description="The final total amount paid")

# 6. Spatial Grouping Algorithm (From Phase 3)
def group_text_into_lines(ocr_results, y_threshold=10):
    items = [{'text': text, 'y_center': (bbox[0][1] + bbox[2][1]) / 2, 'x_start': bbox[0][0]} 
             for bbox, text, conf in ocr_results]
    
    # Sort everything top to bottom
    items.sort(key=lambda item: item['y_center'])
    
    lines, current_line = [], []
    for item in items:
        if not current_line:
            current_line.append(item)
        else:
            line_y_average = sum([i['y_center'] for i in current_line]) / len(current_line)
            # Check if text block belongs on the same horizontal plane
            if abs(item['y_center'] - line_y_average) <= y_threshold:
                current_line.append(item)
            else:
                lines.append(current_line)
                current_line = [item]
    if current_line: 
        lines.append(current_line)
        
    # Sort each row left to right so prices line up correctly after text
    final_text_lines = []
    for line in lines:
        line.sort(key=lambda item: item['x_start'])
        final_text_lines.append(" ".join([item['text'] for item in line]))
        
    return final_text_lines

# 7. The Core API Endpoint
@app.post("/scan-receipt/", response_model=SplitzzaReceipt)
async def scan_receipt_api(file: UploadFile = File(...)):
    """
    Accepts a raw receipt image binary upload, executes OCR, stabilizes layout structures,
    and handles semantic entity extraction via Gemini 2.5 Flash.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")

    try:
        # Step A: Decode raw incoming network bytes into an OpenCV matrix image
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to parse image file data.")

        # Step B: Compute spatial coordinates and extract raw text blobs via EasyOCR
        raw_ocr_data = reader.readtext(image)
        
        # Step C: Normalize coordinate variations to group pieces into clean rows
        structured_lines = group_text_into_lines(raw_ocr_data, y_threshold=10)
        document_text = "\n".join(structured_lines)
        
        # Step D: Construct LLM prompt context
        prompt = f"""
        You are a highly accurate receipt parsing AI for a bill-splitting app.
        Take the following messy OCR text from a restaurant receipt and extract the data structure.
        Fix obvious spelling mistakes in food items and normalize text.
        
        Receipt Text:
        {document_text}
        """

        # Step E: Query Gemini with structural enforcement configurations
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SplitzzaReceipt, 
                temperature=0.0 
            ),
        )
        
        # Step F: Transform response text string safely into native dictionary object for FastAPI output routing
        return json.loads(response.text)

    except Exception as e:
        print(f"🚨 API Runtime Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal AI processing pipeline failure: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # This tells Python to start the server on port 8000 and stay alive
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)