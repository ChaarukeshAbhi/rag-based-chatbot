from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import traceback
from fastapi.responses import HTMLResponse

from utils.llm import LlamaModel
from utils.embedding_faiss import EmbeddingManager
from utils.asr import ASRProcessor
from utils.nlu import NLUProcessor
from utils.calendar import CalendarManager
from utils.mood import MoodTracker
from utils.image import ImageProcessor

app = FastAPI(title="HR RAG Chatbot")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
print("Loading models...")
llm = LlamaModel()
embedding_manager = EmbeddingManager()
asr_processor = ASRProcessor()
nlu_processor = NLUProcessor()
calendar_manager = CalendarManager()
mood_tracker = MoodTracker()
image_processor = ImageProcessor()

# Load FAISS index
try:
    embedding_manager.load_index()
    print("FAISS index loaded successfully")
except Exception as e:
    print(f"Warning: Could not load FAISS index: {e}")
    print("Please run ingest.py to build the index first")

# Request models
class TextQuery(BaseModel):
    text: str
    employee_id: Optional[str] = "default_user"
    language: Optional[str] = None

class MeetingRequest(BaseModel):
    employee_id: str
    title: str
    date: str
    time: str
    duration: Optional[int] = 60
    attendees: Optional[list] = None

class MoodLog(BaseModel):
    employee_id: str
    mood: Optional[str] = None
    text: Optional[str] = None
    rating: Optional[int] = None

# Helper function to clean LLM responses
def clean_response(response):
    """Clean up LLM response to be more chatbot-like"""
    
    try:
        # Remove system prompts and unwanted markers
        lines_to_remove = [
            'system', 'You are a helpful', 'Context:', 'Question:', 
            'user', 'assistant', 'Answer:', '<|', '|>', 'Based on this information',
            'Information:'
        ]
        
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip lines that contain unwanted markers
            if not any(marker in line for marker in lines_to_remove):
                if line:  # Only add non-empty lines
                    cleaned_lines.append(line)
        
        response = ' '.join(cleaned_lines)
        
        # Remove duplicate spaces
        response = ' '.join(response.split())
        
        # Keep only first 3-4 sentences for conciseness
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if len(sentences) > 4:
            response = '. '.join(sentences[:4]) + '.'
        elif sentences:
            response = '. '.join(sentences)
            if not response.endswith('.'):
                response += '.'
        
        return response.strip()
    
    except Exception as e:
        print(f"Error in clean_response: {e}")
        return response

@app.post("/query/text")
async def process_text_query(query: TextQuery):
    """Process text query"""
    try:
        print(f"\n=== Processing Query ===")
        print(f"Text: {query.text}")
        print(f"Employee ID: {query.employee_id}")
        
        # Process query through NLU
        print("Step 1: NLU Processing...")
        nlu_result = nlu_processor.process_query(query.text, query.language)
        print(f"NLU Result - Language: {nlu_result['source_language']}, Intent: {nlu_result['intent']}")
        print(f"English Text: {nlu_result['english_text']}")
        
        # Get relevant context from vector store
        print("Step 2: Vector Search...")
        try:
            search_results = embedding_manager.search(nlu_result['english_text'], k=3)
            context = "\n\n".join([result['document'] for result in search_results])
            print(f"Context retrieved: {len(context)} characters")
        except Exception as e:
            print(f"Vector search error: {e}")
            context = ""
        
        # Handle specific intents
        intent = nlu_result['intent']
        
        # If mood-related, log it
        if intent == 'mood' or mood_tracker.detect_mood(query.text) != 'neutral':
            print("Step 3: Logging mood...")
            mood_tracker.log_mood(
                employee_id=query.employee_id,
                text=query.text
            )
        
        # Generate response
        print("Step 4: Generating response...")
        response = llm.generate_response(
            prompt=nlu_result['english_text'],
            context=context
        )
        print(f"Raw response: {response[:200]}...")

        # Clean up response
        print("Step 5: Cleaning response...")
        response = clean_response(response)
        print(f"Cleaned response: {response}")
        
        # Translate response back if needed
        print("Step 6: Translation check...")
        if nlu_result['source_language'] != 'en':
            print(f"Translating to {nlu_result['source_language']}...")
            translated_response = nlu_processor.translate_from_english(
                response,
                nlu_result['source_language']
            )
            print(f"Translated: {translated_response}")
        else:
            translated_response = response
        
        print("=== Query Complete ===\n")
        
        return {
            "success": True,
            "response": translated_response,
            "original_response": response,
            "intent": intent,
            "language": nlu_result['source_language'],
            "entities": nlu_result['entities']
        }
    
    except Exception as e:
        print(f"\n=== ERROR ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"Full traceback:")
        traceback.print_exc()
        print("=== END ERROR ===\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/voice")
async def process_voice_query(
    audio: UploadFile = File(...),
    employee_id: str = Form("default_user"),
    language: Optional[str] = Form(None)
):
    """Process voice query"""
    try:
        # Save audio temporarily
        temp_path = f"temp_{audio.filename}"
        with open(temp_path, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        # Transcribe audio
        transcription = asr_processor.transcribe_audio(temp_path, language)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if not transcription['success']:
            raise HTTPException(
                status_code=400,
                detail=f"Transcription failed: {transcription.get('error', 'Unknown error')}"
            )
        
        # Process the transcribed text
        query = TextQuery(
            text=transcription['text'],
            employee_id=employee_id,
            language=transcription['language']
        )
        
        result = await process_text_query(query)
        result['transcription'] = transcription['text']
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/meeting/book")
async def book_meeting(meeting: MeetingRequest):
    """Book a meeting"""
    try:
        result = calendar_manager.book_meeting(
            employee_id=meeting.employee_id,
            title=meeting.title,
            date=meeting.date,
            time=meeting.time,
            duration=meeting.duration,
            attendees=meeting.attendees
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/meeting/list/{employee_id}")
async def get_meetings(employee_id: str, date: Optional[str] = None):
    """Get meetings for an employee"""
    try:
        meetings = calendar_manager.get_meetings(employee_id, date)
        return {"success": True, "meetings": meetings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/meeting/cancel/{meeting_id}")
async def cancel_meeting(meeting_id: str):
    """Cancel a meeting"""
    try:
        result = calendar_manager.cancel_meeting(meeting_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/meeting/reschedule/{meeting_id}")
async def reschedule_meeting(meeting_id: str, new_date: str, new_time: str):
    """Reschedule a meeting"""
    try:
        result = calendar_manager.reschedule_meeting(meeting_id, new_date, new_time)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/meeting/availability/{employee_id}")
async def check_availability(employee_id: str, date: str, time: str):
    """Check employee availability"""
    try:
        result = calendar_manager.check_availability(employee_id, date, time)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mood/log")
async def log_mood(mood_log: MoodLog):
    """Log employee mood"""
    try:
        result = mood_tracker.log_mood(
            employee_id=mood_log.employee_id,
            mood=mood_log.mood,
            text=mood_log.text,
            rating=mood_log.rating
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mood/history/{employee_id}")
async def get_mood_history(employee_id: str, days: int = 7):
    """Get mood history for an employee"""
    try:
        history = mood_tracker.get_mood_history(employee_id, days)
        return {"success": True, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mood/analysis/{employee_id}")
async def analyze_mood(employee_id: str, days: int = 30):
    """Analyze mood trends for an employee"""
    try:
        analysis = mood_tracker.analyze_mood_trends(employee_id, days)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/welfare/alerts")
async def get_welfare_alerts():
    """Get welfare alerts for employees"""
    try:
        alerts = mood_tracker.get_welfare_alerts()
        return {"success": True, "alerts": alerts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/document/extract")
async def extract_from_image(image: UploadFile = File(...)):
    """Extract text from uploaded image"""
    try:
        content = await image.read()
        result = image_processor.extract_text_from_image(image_bytes=content)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def serve_login():
    with open("login.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/role-select", response_class=HTMLResponse)
async def serve_role_select():
    with open("role_select.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/chat", response_class=HTMLResponse)
async def serve_chat():
    with open("chat.html", "r", encoding="utf-8") as f:
        return f.read()
    
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)