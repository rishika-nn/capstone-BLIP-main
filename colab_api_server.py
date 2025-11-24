"""
FastAPI Server for Video Search Engine (Colab + ngrok)
Run this in your Google Colab notebook to expose the video search pipeline via API

Installation in Colab:
    !pip install fastapi uvicorn pyngrok nest-asyncio python-multipart

Usage:
    1. Run this script in Colab
    2. Copy the ngrok URL that appears
    3. Use that URL in your React app to connect to the Colab backend
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import uuid
from datetime import datetime, date
import time
import asyncio
import nest_asyncio

# Import your video search engine
from video_search_engine import VideoSearchEngine

# Apply nest_asyncio to allow nested event loops in Colab
nest_asyncio.apply()

# Create FastAPI app
app = FastAPI(
    title="Video Search API (Colab)",
    version="1.0.0",
    description="Video Search Engine powered by BLIP & Pinecone running on Colab GPU"
)

# CORS configuration - Allow requests from React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (ngrok is public anyway)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine = None

# Job tracking for async video processing
processing_jobs = {}

# Create uploads directory
UPLOAD_DIR = "/content/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ==================== Pydantic Models ====================

class ProcessVideoRequest(BaseModel):
    """Request model for video processing"""
    video_name: Optional[str] = None
    video_date: Optional[str] = None
    save_frames: bool = False
    upload_to_pinecone: bool = True
    use_object_detection: bool = False


class SearchRequest(BaseModel):
    """Request model for search"""
    query: str
    top_k: int = 10
    similarity_threshold: float = 0.5
    video_filter: Optional[str] = None
    date_filter: Optional[str] = None
    namespace_filter: Optional[str] = None


class DateRangeSearchRequest(BaseModel):
    """Request model for date range search"""
    query: str
    start_date: str
    end_date: str
    top_k: int = 10
    similarity_threshold: float = 0.5
    namespace_filter: Optional[str] = None


class BatchSearchRequest(BaseModel):
    """Request model for batch search"""
    queries: List[str]
    top_k: int = 5


# ==================== Startup/Shutdown Events ====================

@app.on_event("startup")
async def startup_event():
    """Initialize the Video Search Engine on startup"""
    global engine
    try:
        print("🚀 Initializing Video Search Engine...")
        engine = VideoSearchEngine()
        print("✅ Video Search Engine initialized successfully")
        print("📊 GPU Status:")
        
        import torch
        if torch.cuda.is_available():
            print(f"   ✓ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"   ✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("   ⚠ Running on CPU (slower)")
            
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    global engine
    if engine:
        engine.cleanup()
        print("✅ Resources cleaned up")


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Video Search API",
        "version": "1.0.0",
        "status": "running",
        "engine_ready": engine is not None,
        "endpoints": {
            "health": "/api/health",
            "upload": "/api/upload",
            "process": "/api/process",
            "search": "/api/search",
            "stats": "/api/stats"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    import torch
    
    return {
        "status": "healthy",
        "engine_initialized": engine is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/stats")
async def get_stats():
    """Get Pinecone index statistics"""
    try:
        if not engine:
            raise HTTPException(status_code=500, detail="Engine not initialized")
        
        stats = engine.get_index_stats()
        
        # Convert to JSON-serializable format
        serializable_stats = {
            "index_name": stats.get("index_name", "N/A"),
            "total_vectors": stats.get("total_vectors", 0),
            "dimension": stats.get("dimension", "N/A"),
            "metric": stats.get("metric", "N/A"),
            "capacity": stats.get("capacity", "N/A"),
            "namespaces": {}
        }
        
        # Convert namespace objects to simple dict
        if "namespaces" in stats and stats["namespaces"]:
            for namespace, ns_info in stats["namespaces"].items():
                # Handle NamespaceSummary object - extract vector_count
                if hasattr(ns_info, 'vector_count'):
                    serializable_stats["namespaces"][namespace] = ns_info.vector_count
                elif isinstance(ns_info, int):
                    serializable_stats["namespaces"][namespace] = ns_info
                else:
                    serializable_stats["namespaces"][namespace] = 0
        
        return serializable_stats
        
    except Exception as e:
        print(f"❌ Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dates")
async def get_available_dates():
    """Get list of available dates with videos"""
    try:
        if not engine:
            raise HTTPException(status_code=500, detail="Engine not initialized")
        
        dates = engine.get_available_dates()
        return {"dates": dates, "count": len(dates)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file to Colab
    
    The file is saved to /content/uploads/ with a unique filename
    """
    try:
        # Validate file type
        allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Generate unique filename
        unique_id = str(uuid.uuid4())[:8]
        unique_filename = f"{unique_id}_{file.filename}"
        upload_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save uploaded file
        print(f"📤 Uploading: {file.filename} -> {unique_filename}")
        
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        file_size_mb = len(content) / (1024 * 1024)
        
        print(f"✅ Upload complete: {file_size_mb:.2f} MB")
        
        return {
            "success": True,
            "filename": unique_filename,
            "original_filename": file.filename,
            "path": upload_path,
            "size_mb": round(file_size_mb, 2),
            "message": f"File uploaded successfully ({file_size_mb:.2f} MB)"
        }
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


async def process_video_task(job_id: str, video_path: str, request: ProcessVideoRequest):
    """Background task for video processing"""
    try:
        print(f"\n{'='*80}")
        print(f"🎬 Processing Job: {job_id}")
        print(f"   Video: {video_path}")
        print(f"   Settings: {request.dict()}")
        print(f"{'='*80}\n")
        
        processing_jobs[job_id]["status"] = "processing"
        processing_jobs[job_id]["progress"] = "Extracting frames..."
        
        start_time = time.time()
        
        # Process video with the engine
        stats = engine.process_video(
            video_path=video_path,
            video_name=request.video_name,
            video_date=request.video_date,
            save_frames=request.save_frames,
            upload_to_pinecone=request.upload_to_pinecone,
            use_object_detection=request.use_object_detection
        )
        
        processing_time = time.time() - start_time
        
        # Update job status
        processing_jobs[job_id]["status"] = "completed"
        processing_jobs[job_id]["progress"] = "Complete"
        processing_jobs[job_id]["result"] = stats
        processing_jobs[job_id]["processing_time"] = processing_time
        processing_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
        print(f"\n{'='*80}")
        print(f"✅ Job {job_id} completed in {processing_time:.1f} seconds")
        print(f"   Frames: {stats['frames_with_captions']}")
        print(f"   Embeddings: {stats['embeddings_uploaded']}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ Job {job_id} failed: {e}")
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["error"] = str(e)
        processing_jobs[job_id]["failed_at"] = datetime.now().isoformat()


@app.post("/api/process")
async def process_video(
    background_tasks: BackgroundTasks,
    request: ProcessVideoRequest,
    video_filename: str
):
    """
    Process a video (extract frames, generate captions, create embeddings)
    
    This endpoint starts background processing and returns a job_id
    Use /api/job/{job_id} to check progress
    """
    try:
        if not engine:
            raise HTTPException(status_code=500, detail="Engine not initialized")
        
        # Construct video path
        video_path = os.path.join(UPLOAD_DIR, video_filename)
        
        if not os.path.exists(video_path):
            raise HTTPException(
                status_code=404, 
                detail=f"Video file not found: {video_filename}"
            )
        
        # Set default values
        if not request.video_name:
            request.video_name = os.path.splitext(video_filename)[0]
        
        if not request.video_date:
            request.video_date = date.today().strftime("%Y-%m-%d")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job tracking
        processing_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": "Queued for processing...",
            "video_path": video_path,
            "video_filename": video_filename,
            "request": request.dict(),
            "created_at": datetime.now().isoformat()
        }
        
        # Start background processing
        background_tasks.add_task(
            process_video_task,
            job_id,
            video_path,
            request
        )
        
        print(f"📋 Job {job_id} queued for processing")
        
        return {
            "success": True,
            "job_id": job_id,
            "status": "queued",
            "message": "Video processing started. Use /api/job/{job_id} to check progress."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Process request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    """Check the status of a processing job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_jobs[job_id]


@app.get("/api/jobs")
async def list_jobs():
    """List all processing jobs"""
    return {
        "jobs": list(processing_jobs.values()),
        "total": len(processing_jobs)
    }


@app.post("/api/search")
async def search(request: SearchRequest):
    """
    Search videos using natural language query
    
    Example queries:
    - "person walking"
    - "black bag on the ground"
    - "someone talking on phone"
    """
    try:
        if not engine:
            raise HTTPException(status_code=500, detail="Engine not initialized")
        
        print(f"🔍 Search query: '{request.query}'")
        if request.date_filter:
            print(f"   Date filter: {request.date_filter}")
        if request.namespace_filter:
            print(f"   Category filter: {request.namespace_filter}")
        
        results = engine.search(
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            video_filter=request.video_filter,
            date_filter=request.date_filter,
            namespace_filter=request.namespace_filter
        )
        
        print(f"✅ Found {len(results)} results")
        
        return {
            "success": True,
            "query": request.query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search/daterange")
async def search_by_date_range(request: DateRangeSearchRequest):
    """Search videos within a date range"""
    try:
        if not engine:
            raise HTTPException(status_code=500, detail="Engine not initialized")
        
        print(f"🔍 Date range search: '{request.query}' ({request.start_date} to {request.end_date})")
        
        results = engine.search_by_date_range(
            query=request.query,
            start_date=request.start_date,
            end_date=request.end_date,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            namespace_filter=request.namespace_filter
        )
        
        print(f"✅ Found {len(results)} results")
        
        return {
            "success": True,
            "query": request.query,
            "date_range": f"{request.start_date} to {request.end_date}",
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search/batch")
async def batch_search(request: BatchSearchRequest):
    """Search multiple queries at once"""
    try:
        if not engine:
            raise HTTPException(status_code=500, detail="Engine not initialized")
        
        print(f"🔍 Batch search: {len(request.queries)} queries")
        
        results = engine.batch_search(
            queries=request.queries,
            top_k=request.top_k
        )
        
        total_results = sum(len(r) for r in results.values())
        print(f"✅ Found {total_results} total results")
        
        return {
            "success": True,
            "results": results,
            "query_count": len(request.queries),
            "total_results": total_results
        }
        
    except Exception as e:
        print(f"❌ Batch search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clear")
async def clear_index():
    """Clear all vectors from Pinecone index (use with caution!)"""
    try:
        if not engine:
            raise HTTPException(status_code=500, detail="Engine not initialized")
        
        print("⚠️ Clearing Pinecone index...")
        success = engine.clear_index()
        
        if success:
            print("✅ Index cleared successfully")
            return {
                "success": True,
                "message": "Pinecone index cleared successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to clear index")
            
    except Exception as e:
        print(f"❌ Clear index failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job from tracking"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del processing_jobs[job_id]
    return {"success": True, "message": f"Job {job_id} deleted"}


# ==================== Helper Functions ====================

def start_server_with_ngrok(port: int = 8000):
    """
    Start the FastAPI server with ngrok tunnel
    
    Usage in Colab:
        from colab_api_server import start_server_with_ngrok
        start_server_with_ngrok()
    """
    try:
        from pyngrok import ngrok
        import uvicorn
        
        # Kill any existing ngrok tunnels
        ngrok.kill()
        
        # Start ngrok tunnel
        print(f"🌐 Starting ngrok tunnel on port {port}...")
        public_url = ngrok.connect(port)
        
        print("\n" + "="*80)
        print("✅ COLAB API SERVER READY")
        print("="*80)
        print(f"\n🌍 Public URL: {public_url}")
        print(f"\n📋 Copy this URL and paste it into your React app (.env file)")
        print(f"   REACT_APP_API_URL={public_url}")
        print("\n" + "="*80)
        print("\n🚀 Starting FastAPI server...")
        print("   Keep this notebook running while using the React app")
        print("   The server will process videos using Colab's GPU")
        print("\n" + "="*80 + "\n")
        
        # Start uvicorn server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
        
    except ImportError:
        print("❌ Error: pyngrok not installed")
        print("   Run: !pip install pyngrok")
    except Exception as e:
        print(f"❌ Error starting server: {e}")


if __name__ == "__main__":
    # For local testing (not in Colab)
    import uvicorn
    print("⚠️ Running in local mode (no ngrok)")
    print("   For Colab, use: start_server_with_ngrok()")
    
    async def start_server():
        config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    # Get or create event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(start_server())
