import os
import json
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai
from google.genai import types

# ---------------------------------------------------------
# 1. INITIALIZATION & SETUP
# ---------------------------------------------------------
app = FastAPI(title="Splitzza AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini Client
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY environment variable is missing!")
client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------
# 2. DATA MODELS (Strict Math & Validation)
# ---------------------------------------------------------
class ReceiptItem(BaseModel):
    name: str = Field(description="Expanded name of the item (e.g. MSL = Masala, S/W = Sandwich)")
    quantity: int = Field(description="Exact quantity ordered.")
    unit_price: float = Field(description="The cost of ONE unit (The 'Rate' column).")
    total_price: float = Field(description="Quantity * unit_price (The 'Amount' column).")

class ReceiptMeta(BaseModel):
    subtotal: Optional[float] = Field(description="The subtotal before taxes", default=0.0)
    tax: Optional[float] = Field(description="Sum of all taxes (SGST + CGST + VAT)", default=0.0)
    total: Optional[float] = Field(description="The grand total paid", default=0.0)
    restaurant_name: Optional[str] = Field(description="The name of the restaurant", default="Unknown")

class SplitzzaReceipt(BaseModel):
    is_valid_receipt: bool = Field(description="True if the image is actually a restaurant/grocery bill. False if it is a random photo.", default=True)
    items: List[ReceiptItem] = Field(description="List of extracted food/drink items", default_factory=list)
    meta: ReceiptMeta = Field(description="Metadata like totals and restaurant name", default_factory=ReceiptMeta)

# ---------------------------------------------------------
# 3. API ENDPOINTS
# ---------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "online", "app": "Splitzza Backend"}

@app.post("/scan-receipt/", response_model=SplitzzaReceipt)
async def scan_receipt_api(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")
    
    try:
        # Read the raw image bytes directly
        image_bytes = await file.read()
        
        # Convert bytes into a Gemini Vision Part object
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=file.content_type)
        
        # Unified, Bulletproof Prompt (Validation + Strict/Forgiving Math)
        prompt = """
        You are an expert data extraction AI specifically trained on Indian restaurant receipts.
        Analyze the provided image and extract the data into the requested JSON schema.
        
        CRITICAL RULES:
        1. IMAGE VALIDATION (GATEKEEPER): Determine if the image is actually a restaurant or grocery receipt. If it is NOT a receipt (e.g., a person, landscape, dog, or random screenshot), set `is_valid_receipt` to false, leave `items` empty, and set all metadata to 0. DO NOT hallucinate data.
        2. ITEM-LEVEL MATH (STRICT): Differentiate between 'Rate' (unit price) and 'Amount' (total item price). Ensure that (quantity * unit_price) exactly equals total_price for every single item.
        3. ABBREVIATIONS: Expand Indian restaurant abbreviations (e.g., 'MSL' -> 'Masala', 'S/W' -> 'Sandwich', 'VDA' -> 'Wada').
        4. RECEIPT-LEVEL MATH (FORGIVING): Indian bills often have mathematical typos (e.g., the printed Subtotal might say 505 even if the items add up to 605). DO NOT panic if the overall subtotal math doesn't perfectly add up. Trust the printed Tax and Grand Total numbers.
        5. EXPLICIT TAX EXTRACTION: Search the bottom of the bill for 'SGST', 'CGST', 'IGST', or 'Service Charge'. You MUST extract the numeric values next to these and ADD them together into the 'tax' field. (e.g., If SGST is 15.12 and CGST is 15.12, the tax field is 30.24).
        6. GRAND TOTAL: Extract the final billed amount (e.g., 635) and put it in the 'total' field. Ignore FSSAI numbers and addresses.
        """

        max_attempts = 4
        base_delay = 3 
        
        for attempt in range(max_attempts):
            try:
                # Pass BOTH the prompt and the image_part to the multimodal model
                response = client.models.generate_content(
                    model='gemini-2.5-flash', 
                    contents=[prompt, image_part],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=SplitzzaReceipt, 
                        temperature=0.0 # Keep at 0 for strict data extraction
                    ),
                )
                return json.loads(response.text)
                
            except Exception as api_error:
                error_message = str(api_error)
                print(f"⚠️ API Error (Attempt {attempt + 1}/{max_attempts}): {error_message}")
                
                if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
                    print("🛑 Rate limit hit. Halting retries.")
                    raise HTTPException(
                        status_code=429, 
                        detail="You have exceeded the free AI scanning limit. Please wait 60 seconds and try again."
                    )
                if attempt < max_attempts - 1:
                    wait_time = base_delay * (2 ** attempt) 
                    time.sleep(wait_time)
                    continue
                
                raise HTTPException(
                    status_code=503, 
                    detail="The AI server is currently congested. Please try again in a minute."
                )

    except HTTPException as http_exc:
        raise http_exc
            
    except Exception as e:
        print(f"🚨 API Runtime Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal AI processing failure: {str(e)}")