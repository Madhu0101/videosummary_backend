import os
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, UploadFile, File, HTTPException, Depends, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from bson import ObjectId
import logging
from datetime import datetime
from db import users_collection, videos_collection, tokens_collection
from models import UserCreate, UserLogin, Token
from controllers import (
    verify_token, get_password_hash, verify_password, create_access_token,
    process_transcription, process_summary_generation, summary_generator, ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE, UPLOAD_DIR, ACCESS_TOKEN_EXPIRE_MINUTES
)

router = APIRouter()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
@router.post("/api/auth/register", response_model=Token)
async def register(user: UserCreate):
    if await users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    if await users_collection.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already taken")

    hashed_password = get_password_hash(user.password)
    user_doc = {
        "username": user.username,
        "email": user.email,
        "password_hash": hashed_password,
        "created_at": str(datetime.now())
    }
    result = await users_collection.insert_one(user_doc)
    user_id = str(result.inserted_id)

    from datetime import timedelta
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_id}, expires_delta=access_token_expires
    )
    await tokens_collection.insert_one({"token": access_token, "user_id": user_id})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user_id,
        "email": user.email,
        "username": user.username
    }

@router.post("/api/auth/login", response_model=Token)
async def login(user: UserLogin):
    user_data = await users_collection.find_one({"email": user.email})
    if not user_data or not verify_password(user.password, user_data["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id = str(user_data["_id"])

    from datetime import timedelta
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_id}, expires_delta=access_token_expires
    )
    await tokens_collection.insert_one({"token": access_token, "user_id": user_id})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user_id,
        "email": user_data["email"],
        "username": user_data["username"]
    }

@router.post("/api/auth/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    token = credentials.credentials
    await tokens_collection.delete_one({"token": token})
    return {"message": "Successfully logged out"}

@router.get("/api/auth/me")
async def get_current_user(user_id: str = Depends(verify_token)):
    user_data = await users_collection.find_one({"_id": ObjectId(user_id)})
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "user_id": user_id,
        "username": user_data["username"],
        "email": user_data["email"],
        "created_at": user_data["created_at"]
    }

@router.post("/api/videos/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Depends(verify_token)
):
    try:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(400, detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(413, detail="File too large (max 2GB)")
        file_id = str(ObjectId())
        file_path = os.path.join(UPLOAD_DIR, f"{user_id}_{file_id}{file_ext}")
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        video_doc = {
            "_id": ObjectId(file_id),
            "original_filename": file.filename,
            "file_path": file_path,
            "file_size": len(content),
            "status": "uploaded",
            "summary_status": "pending",
            "uploaded_at": str(datetime.now()),
            "user_id": user_id
        }
        await videos_collection.insert_one(video_doc)
        background_tasks.add_task(process_transcription, user_id, file_id, file_path)
        return JSONResponse({
            "video_id": file_id,
            "status": "uploaded",
            "message": "Video uploaded and processing started"
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail="Internal server error during upload")

@router.get("/api/videos/{video_id}/status")
async def get_status(video_id: str, user_id: str = Depends(verify_token)):
    video = await videos_collection.find_one({"_id": ObjectId(video_id), "user_id": user_id})
    if not video:
        raise HTTPException(404, detail="Video not found")
    return {
        "video_id": video_id,
        "status": video.get("status"),
        "summary_status": video.get("summary_status", "pending"),
        "transcription": video.get("transcription"),
        "segments": video.get("segments", []),
        "language": video.get("language"),
        "summary": video.get("summary")
    }

@router.get("/api/videos")
async def get_user_videos_list(user_id: str = Depends(verify_token)):
    videos = await videos_collection.find({"user_id": user_id}).to_list(length=1000)
    videos_list = []
    for video in videos:
        videos_list.append({
            "video_id": str(video["_id"]),
            "original_filename": video.get("original_filename"),
            "status": video.get("status"),
            "summary_status": video.get("summary_status", "pending"),
            "uploaded_at": video.get("uploaded_at"),
            "file_size": video.get("file_size"),
            "transcription": video.get("transcription"),
            "summary": video.get("summary"),               
        })
    return {"videos": videos_list}

@router.post("/api/videos/{video_id}/generate-summary")
async def generate_summary(video_id: str, background_tasks: BackgroundTasks, user_id: str = Depends(verify_token)):
    video = await videos_collection.find_one({"_id": ObjectId(video_id), "user_id": user_id})
    if not video:
        raise HTTPException(404, detail="Video not found")
    if video.get("status") != "transcribed":
        raise HTTPException(400, detail="Video must be transcribed first")
    if video.get("summary_status") == "generating":
        return JSONResponse({
            "message": "Summary generation already in progress"
        })
    background_tasks.add_task(process_summary_generation, user_id, video_id)
    return JSONResponse({
        "message": "Summary generation started",
        "video_id": video_id
    })

@router.get("/api/videos/{video_id}/summary")
async def get_summary(video_id: str, user_id: str = Depends(verify_token)):
    video = await videos_collection.find_one({"_id": ObjectId(video_id), "user_id": user_id})
    if not video:
        raise HTTPException(404, detail="Video not found")
    if "summary" not in video:
        raise HTTPException(202, detail="Summary not ready yet")
    return {
        "video_id": video_id,
        "summary_status": video.get("summary_status"),
        "summary": video["summary"],
        "generated_at": video.get("summary_generated_at")
    }

@router.get("/api/videos/{video_id}/play")
async def play_video(video_id: str, user_id: str = Depends(verify_token)):
    video = await videos_collection.find_one({"_id": ObjectId(video_id), "user_id": user_id})
    if not video:
        raise HTTPException(404, detail="Video not found")
    file_path = video["file_path"]
    if not os.path.exists(file_path):
        raise HTTPException(404, detail="Video file not found")
    return FileResponse(
        file_path,
        media_type="video/mp4",
        filename=video["original_filename"]
    )

@router.delete("/api/videos/{video_id}")
async def delete_video(video_id: str, user_id: str = Depends(verify_token)):
    video = await videos_collection.find_one({"_id": ObjectId(video_id), "user_id": user_id})
    if not video:
        raise HTTPException(404, detail="Video not found")
    try:
        file_path = video.get("file_path")
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        pass
    await videos_collection.delete_one({"_id": ObjectId(video_id), "user_id": user_id})
    return {"message": "Video and all related data deleted successfully"}

@router.get("/api/videos/{video_id}/download/transcript/{format}")
async def download_transcript(video_id: str, format: str, user_id: str = Depends(verify_token)):
    video = await videos_collection.find_one({"_id": ObjectId(video_id), "user_id": user_id})
    if not video:
        raise HTTPException(404, detail="Video not found")
    if "transcription" not in video:
        raise HTTPException(400, detail="Transcript not available yet")
    transcript = video["transcription"]
    filename = f"{video_id}_transcript"
    try:
        if format == "txt":
            buffer = summary_generator.generate_text_file(transcript)
            return StreamingResponse(
                buffer,
                media_type="text/plain",
                headers={"Content-Disposition": f"attachment; filename={filename}.txt"}
            )
        elif format == "docx":
            buffer = summary_generator.generate_docx_file(transcript)
            return StreamingResponse(
                buffer,
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={"Content-Disposition": f"attachment; filename={filename}.docx"}
            )
        elif format == "pdf":
            buffer = summary_generator.generate_pdf_file(transcript)
            return StreamingResponse(
                buffer,
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename={filename}.pdf"}
            )
        else:
            raise HTTPException(400, detail="Unsupported format. Use txt, docx, or pdf")
    except Exception as e:
        raise HTTPException(500, detail="Failed to generate download file")

@router.get("/api/videos/{video_id}/download/summary/{format}")
async def download_summary(
    video_id: str, 
    format: str,
    user_id: str = Depends(verify_token)
):
    video = await videos_collection.find_one({"_id": ObjectId(video_id), "user_id": user_id})
    if not video:
        raise HTTPException(404, detail="Video not found")
    if "summary" not in video:
        raise HTTPException(400, detail="Summary not available yet")
    summary_data = video["summary"]
    filename = f"{video['original_filename']}_summary"
    content = summary_data["paragraph_summary"] + "\n\n"
    content += "\n".join(f"- {point}" for point in summary_data["bullet_points"]) + "\n\n"
    content += "\n".join(
        f"{hl['formatted_time']}: {hl['text']}"
        for hl in summary_data["key_highlights"]
    )
    try:
        if format == "txt":
            buffer = summary_generator.generate_text_file(content)
            return StreamingResponse(
                buffer,
                media_type="text/plain",
                headers={"Content-Disposition": f"attachment; filename={filename}.txt"}
            )
        elif format == "docx":
            buffer = summary_generator.generate_docx_file(content)
            return StreamingResponse(
                buffer,
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={"Content-Disposition": f"attachment; filename={filename}.docx"}
            )
        elif format == "pdf":
            buffer = summary_generator.generate_pdf_file(content)
            return StreamingResponse(
                buffer,
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename={filename}.pdf"}
            )
        else:
            raise HTTPException(400, detail="Unsupported format. Use txt, docx, or pdf")
    except Exception as e:
        logger.error(f"Failed to generate summary download: {e}")
        raise HTTPException(500, detail="Failed to generate download file")