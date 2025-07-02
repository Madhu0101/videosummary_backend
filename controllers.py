import os
import ssl
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uuid
from pathlib import Path
import whisper
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from io import BytesIO
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import zipfile
import jwt
from passlib.context import CryptContext
from bson import ObjectId
from db import users_collection, videos_collection, tokens_collection
from models import UserCreate, UserLogin
from models import UserCreate, UserLogin, Token
# Disable SSL verification (temporary workaround)
ssl._create_default_https_context = ssl._create_unverified_context

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# JWT Configuration
SECRET_KEY = "your-secret-key-change-in-production"  # Change this in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 7 * 24 * 60 

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()

# Configuration
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB
MODEL_CACHE_DIR = "./model_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

def str_objectid(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    return obj

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    token_doc = await tokens_collection.find_one({"token": token})
    if not token_doc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_user_videos(user_id: str) -> Dict:
    videos = await videos_collection.find({"user_id": user_id}).to_list(length=1000)
    return {str(v["_id"]): v for v in videos}

class SummaryGenerator:
    def __init__(self):
        self.summarizer = None
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        if not self.summarizer:
            try:
                logger.info("Loading T5-small model...")
                model_name = "t5-small"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.summarizer = pipeline(
                    "summarization",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("Tiny model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load tiny model: {e}")
                self.summarizer = lambda x: [{"summary_text": " ".join(x.split()[:50])}]
        return self.summarizer
    
    def chunk_text(self, text: str, max_chunk_length: int = 1024) -> List[str]:
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_chunk) + len(sentence) < max_chunk_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    
    def extract_key_points(self, text: str) -> List[str]:
        key_indicators = [
            r'important[ly]?', r'key', r'main', r'primary', r'essential',
            r'decided?', r'agreed?', r'concluded?', r'action', r'next steps?',
            r'deadline', r'by \d+', r'must', r'should', r'will', r'need to',
            r'problem', r'issue', r'challenge', r'solution', r'recommend'
        ]
        sentences = re.split(r'[.!?]+', text)
        key_points = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            for indicator in key_indicators:
                if re.search(indicator, sentence, re.IGNORECASE):
                    key_points.append(sentence)
                    break
        return key_points[:10]
    
    def find_highlight_timestamps(self, segments: List[Dict], key_points: List[str]) -> List[Dict]:
        highlights = []
        for key_point in key_points:
            words = key_point.lower().split()[:5]
            for segment in segments:
                segment_text = segment.get('text', '').lower()
                matches = sum(1 for word in words if word in segment_text)
                if matches >= min(3, len(words) * 0.6):
                    highlights.append({
                        "timestamp": segment.get('start', 0),
                        "end_timestamp": segment.get('end', 0),
                        "text": key_point,
                        "formatted_time": self.format_timestamp(segment.get('start', 0))
                    })
                    break
        return highlights
    
    def format_timestamp(self, seconds: float) -> str:
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def generate_summary(self, transcript: str, segments: List[Dict]) -> Dict:
        try:
            summarizer = self.load_model()
            chunks = self.chunk_text(transcript)
            paragraph_summaries = []
            for chunk in chunks:
                words = chunk.split()
                if len(words) < 20:
                    continue
                # Set max_length to be less than input length
                max_length = min(150, int(len(words) * 0.7))
                min_length = min(50, int(len(words) * 0.3))
                try:
                    summary = summarizer(
                        chunk,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False
                    )
                    paragraph_summaries.append(summary[0]['summary_text'])
                except Exception as e:
                    logger.warning(f"Failed to summarize chunk: {e}")
                    continue
            combined_summary = " ".join(paragraph_summaries)
            key_points = self.extract_key_points(transcript)
            bullet_points = []
            for point in key_points:
                clean_point = point.strip().rstrip('.')
                if clean_point and len(clean_point) > 10:
                    bullet_points.append(clean_point)
            highlights = self.find_highlight_timestamps(segments, key_points)
            return {
                "paragraph_summary": combined_summary,
                "bullet_points": bullet_points,
                "key_highlights": highlights,
                "summary_stats": {
                    "original_length": len(transcript.split()),
                    "summary_length": len(combined_summary.split()),
                    "compression_ratio": round(len(combined_summary.split()) / len(transcript.split()), 2) if len(transcript.split()) else 0,
                    "key_points_count": len(bullet_points),
                    "highlights_count": len(highlights)
                }
            }
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return {
                "paragraph_summary": "Summary generation failed. Please try again.",
                "bullet_points": [],
                "key_highlights": [],
                "error": str(e)
            }

    def generate_text_file(self, content: str) -> BytesIO:
        buffer = BytesIO()
        buffer.write(content.encode('utf-8'))
        buffer.seek(0)
        return buffer

    def generate_docx_file(self, content: str) -> BytesIO:
        doc = Document()
        doc.add_paragraph(content)
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer

    def generate_pdf_file(self, content: str) -> BytesIO:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [Paragraph(content, styles["Normal"])]
        doc.build(story)
        buffer.seek(0)
        return buffer

    def generate_zip_file(self, files: Dict[str, BytesIO]) -> BytesIO:
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for filename, file_buffer in files.items():
                zip_file.writestr(filename, file_buffer.getvalue())
        buffer.seek(0)
        return buffer

class VideoTranscriber:
    def __init__(self):
        self.model = None
        
    def load_model(self):
        if not self.model:
            logger.info("Loading Whisper model...")
            self.model = whisper.load_model("base")
        return self.model
    
    def transcribe(self, audio_path: str) -> Dict:
        model = self.load_model()
        result = model.transcribe(audio_path, verbose=True)
        return {
            "text": result["text"],
            "segments": result["segments"],
            "language": result["language"]
        }

transcriber = VideoTranscriber()
summary_generator = SummaryGenerator()

async def process_transcription(user_id: str, video_id: str, file_path: str):
    try:
        logger.info(f"Starting transcription for video {video_id} (user: {user_id})")
        await videos_collection.update_one(
            {"_id": ObjectId(video_id), "user_id": user_id},
            {"$set": {"status": "transcribing"}}
        )
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")
        result = transcriber.transcribe(file_path)
        await videos_collection.update_one(
            {"_id": ObjectId(video_id), "user_id": user_id},
            {"$set": {
                "status": "transcribed",
                "transcription": result["text"],
                "segments": result["segments"],
                "language": result["language"],
                "completed_at": str(datetime.now())
            }}
        )
        logger.info(f"Successfully transcribed video {video_id} for user {user_id}")
    except Exception as e:
        error_msg = f"Transcription failed: {str(e)}"
        logger.error(error_msg)
        await videos_collection.update_one(
            {"_id": ObjectId(video_id), "user_id": user_id},
            {"$set": {"status": error_msg}}
        )

async def process_summary_generation(user_id: str, video_id: str):
    try:
        logger.info(f"Starting summary generation for video {video_id} (user: {user_id})")
        await videos_collection.update_one(
            {"_id": ObjectId(video_id), "user_id": user_id},
            {"$set": {"summary_status": "generating"}}
        )
        video = await videos_collection.find_one({"_id": ObjectId(video_id), "user_id": user_id})
        transcript = video.get("transcription", "")
        segments = video.get("segments", [])
        summary_result = summary_generator.generate_summary(transcript, segments)
        await videos_collection.update_one(
            {"_id": ObjectId(video_id), "user_id": user_id},
            {"$set": {
                "summary_status": "completed",
                "summary": summary_result,
                "summary_generated_at": str(datetime.now())
            }}
        )
        logger.info(f"Successfully generated summary for video {video_id} (user: {user_id})")
    except Exception as e:
        error_msg = f"Summary generation failed: {str(e)}"
        logger.error(error_msg)
        await videos_collection.update_one(
            {"_id": ObjectId(video_id), "user_id": user_id},
            {"$set": {"summary_status": error_msg}}
        )
