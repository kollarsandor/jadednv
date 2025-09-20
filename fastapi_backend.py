# =======================================================================

#!/usr/bin/env python3
"""
FastAPI Backend Server for JADED AI Platform
Integrates with quantum_integration.py and unified_integration_app.py
Handles missing imports with fallbacks
"""

import os
import json
import asyncio
import logging
import subprocess
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# FastAPI imports
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import with fallbacks - Production Implementation
try:
    from quantum_integration import QuantumProteinFolder
    QUANTUM_AVAILABLE = True
    logger.info("✅ Quantum integration available")
except ImportError as e:
    QUANTUM_AVAILABLE = False
    logger.warning(f"⚠️ Quantum integration not available: {e}")

    # Create fallback class for type checking
    class QuantumProteinFolder:
        def __init__(self):
            pass
        async def fold_protein_async(self, sequence: str, backend: str = "ibm_torino") -> Dict:
            return {"status": "fallback", "message": "Quantum integration not available"}

try:
    from unified_integration_app import UnifiedIntegrationApp
    UNIFIED_APP_AVAILABLE = True
    logger.info("✅ Unified integration app available")
except ImportError as e:
    UNIFIED_APP_AVAILABLE = False
    logger.warning(f"⚠️ Unified integration app not available: {e}")

    # Create fallback class for type checking
    class UnifiedIntegrationApp:
        def __init__(self):
            self.services_status = {}
        def ai_text_generation_demo(self, prompt: str, model: Optional[str] = None) -> str:
            return "Unified integration app not available"

try:
    from integrations import UnifiedClient
    INTEGRATIONS_AVAILABLE = True
    logger.info("✅ Integrations module available")
except ImportError as e:
    INTEGRATIONS_AVAILABLE = False
    logger.warning(f"⚠️ Integrations module not available: {e}")

    # Create fallback class for type checking
    class UnifiedClient:
        def __init__(self):
            pass
        @property
        def nvidia(self):
            return self
        def generate_text(self, model: str, prompt: str, max_tokens: int, temperature: float) -> Dict:
            return {"choices": [{"message": {"content": "UnifiedClient not available"}}]}

# Pydantic models for API requests
class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000

class ProteinFoldingRequest(BaseModel):
    sequence: str
    backend: Optional[str] = "ibm_torino"
    method: Optional[str] = "quantum"

class VectorSearchRequest(BaseModel):
    query_text: str
    documents: List[str]
    top_k: Optional[int] = 3

class ServiceStatusRequest(BaseModel):
    service_name: Optional[str] = None

class CerebrasMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str

class CerebrasRequest(BaseModel):
    messages: List[CerebrasMessage]
    model: Optional[str] = "gpt-oss-120b"
    stream: Optional[bool] = True
    max_completion_tokens: Optional[int] = 65536
    temperature: Optional[float] = 0.24
    top_p: Optional[float] = 0.18
    reasoning_effort: Optional[str] = "low"
    stop: Optional[List[str]] = None

# Initialize FastAPI app
app = FastAPI(
    title="JADED AI Platform Backend",
    description="Backend server for the JADED Deep Discovery AI Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized later with error handling)
quantum_folder = None
unified_app = None
unified_client = None

def initialize_services():
    """Initialize services with proper error handling"""
    global quantum_folder, unified_app, unified_client

    # Initialize Quantum Protein Folder
    if QUANTUM_AVAILABLE:
        try:
            quantum_folder = QuantumProteinFolder()
            logger.info("✅ Quantum protein folder initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize quantum folder: {e}")

    # Initialize Unified Integration App
    if UNIFIED_APP_AVAILABLE:
        try:
            unified_app = UnifiedIntegrationApp()
            logger.info("✅ Unified integration app initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize unified app: {e}")

    # Initialize Unified Client
    if INTEGRATIONS_AVAILABLE:
        try:
            unified_client = UnifiedClient()
            logger.info("✅ Unified client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize unified client: {e}")

# Initialize services at startup
@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("🚀 Starting JADED AI Platform Backend")
    initialize_services()
    logger.info("✅ Backend initialization complete")

# Serve static files (index.html and assets)
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main index.html file"""
    try:
        index_path = Path("index.html")
        if index_path.exists():
            return FileResponse("index.html", media_type="text/html")
        else:
            # Fallback HTML if index.html is missing
            fallback_html = """
            <!DOCTYPE html>
            <html lang="hu">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>JADED AI Platform - Backend Active</title>
                <style>
                    body { font-family: Inter, sans-serif; background: #000; color: #f0f0f0; margin: 0; padding: 2rem; }
                    .container { max-width: 800px; margin: 0 auto; text-align: center; }
                    h1 { color: #0A61F7; margin-bottom: 1rem; }
                    .status { background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px; margin: 1rem 0; }
                    .available { color: #39FF14; }
                    .unavailable { color: #ff4444; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🧬 JADED AI Platform Backend</h1>
                    <div class="status">
                        <h2>Backend Status: Active</h2>
                        <p>FastAPI server is running on port 5000</p>
                        <p><a href="/docs" style="color: #0A61F7;">API Documentation</a></p>
                    </div>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=fallback_html)
    except Exception as e:
        logger.error(f"Error serving index: {e}")
        return HTMLResponse(content=f"<h1>Error: {e}</h1>", status_code=500)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "quantum_available": QUANTUM_AVAILABLE,
            "unified_app_available": UNIFIED_APP_AVAILABLE,
            "integrations_available": INTEGRATIONS_AVAILABLE
        }
    }

# API Endpoints
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat with AI models"""
    try:
        if unified_app and hasattr(unified_app, 'ai_text_generation_demo'):
            result = unified_app.ai_text_generation_demo(
                prompt=request.message,
                model=request.model
            )
            return {"status": "success", "response": result}
        elif unified_client and hasattr(unified_client, 'nvidia'):
            # Direct NVIDIA client call
            try:
                result = unified_client.nvidia.generate_text(
                    model=request.model or 'meta/llama-3.1-8b-instruct',
                    prompt=request.message,
                    max_tokens=request.max_tokens or 1000,
                    temperature=request.temperature or 0.7
                )
                return {
                    "status": "success",
                    "response": result['choices'][0]['message']['content'],
                    "model": request.model,
                    "usage": result.get('usage', {})
                }
            except Exception as e:
                logger.error(f"NVIDIA API error: {e}")
                raise HTTPException(status_code=500, detail=f"AI service error: {e}")
        else:
            # Fallback response
            fallback_responses = [
                f"I understand you're asking: '{request.message}'. The AI services are currently in simulation mode.",
                f"Your message '{request.message}' has been received. The full AI capabilities will be available when API keys are configured.",
                f"Processing your query: '{request.message}'. This is a fallback response as the AI backend is in development mode."
            ]
            import random
            response = random.choice(fallback_responses)
            return {
                "status": "fallback",
                "response": response,
                "message": "AI services running in simulation mode"
            }
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat service error: {e}")

@app.get("/api/services")
async def get_services_status():
    """Get status of all integrated services"""
    try:
        if unified_app and hasattr(unified_app, 'services_status'):
            return {
                "status": "success",
                "services": unified_app.services_status,
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Fallback service status
            fallback_services = {
                "nvidia": {"status": "SIMULATION", "capabilities": ["text_generation", "fallback_mode"]},
                "quantum": {"status": "SIMULATION", "capabilities": ["protein_folding", "fallback_mode"]},
                "vector_search": {"status": "SIMULATION", "capabilities": ["similarity_search", "fallback_mode"]},
                "cache": {"status": "SIMULATION", "capabilities": ["memory_cache", "fallback_mode"]}
            }
            return {
                "status": "fallback",
                "services": fallback_services,
                "message": "Services running in simulation mode",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Services status error: {e}")
        raise HTTPException(status_code=500, detail=f"Services status error: {e}")

@app.post("/api/quantum")
async def quantum_protein_folding(request: ProteinFoldingRequest):
    """Quantum-enhanced protein folding endpoint"""
    try:
        if quantum_folder:
            result = await quantum_folder.fold_protein_async(
                sequence=request.sequence,
                backend=request.backend or "ibm_torino"
            )
            return {"status": "success", "result": result}
        else:
            # Fallback protein folding simulation
            logger.info(f"Quantum fallback for sequence length: {len(request.sequence)}")

            # Simple fallback coordinates
            coordinates = []
            for i, aa in enumerate(request.sequence[:10]):  # First 10 residues
                x = i * 3.8 + (hash(aa) % 100) / 100.0
                y = (hash(aa + str(i)) % 200) / 100.0 - 1.0
                z = (hash(str(i) + aa) % 200) / 100.0 - 1.0
                coordinates.append([round(x, 3), round(y, 3), round(z, 3)])

            return {
                "status": "fallback",
                "result": {
                    "method": "classical_simulation",
                    "coordinates": coordinates,
                    "total_residues": len(request.sequence),
                    "message": "Quantum services running in simulation mode"
                }
            }
    except Exception as e:
        logger.error(f"Quantum endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Quantum service error: {e}")

@app.post("/api/cerebras")
async def cerebras_chat_completion(request: CerebrasRequest):
    """Cerebras AI chat completion endpoint using Zig client"""
    try:
        # Check for CEREBRAS_API_KEY
        if not os.getenv("CEREBRAS_API_KEY"):
            logger.warning("CEREBRAS_API_KEY not found, using fallback mode")
            # Fallback response when API key is not available
            sample_responses = [
                "I'm a Cerebras AI simulation. The actual API key is not configured.",
                "This is a fallback response from the Cerebras integration. Please set CEREBRAS_API_KEY environment variable.",
                "Cerebras AI client is ready but needs API key configuration to work with the real service."
            ]
            import random
            return {
                "status": "fallback",
                "response": random.choice(sample_responses),
                "model": request.model,
                "message": "CEREBRAS_API_KEY not configured, running in simulation mode"
            }

        # Check if Zig client binary exists
        zig_binary_path = Path("cerebras_client").resolve()
        if not zig_binary_path.exists():
            logger.info("Compiling Zig Cerebras client...")
            try:
                # Compile Zig client
                compile_result = subprocess.run(
                    ["zig", "build-exe", "cerebras_client.zig", "-O", "ReleaseFast"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if compile_result.returncode != 0:
                    logger.error(f"Zig compilation failed: {compile_result.stderr}")
                    raise HTTPException(status_code=500, detail="Failed to compile Zig client")
                logger.info("✅ Zig client compiled successfully")
            except subprocess.TimeoutExpired:
                logger.error("Zig compilation timed out")
                raise HTTPException(status_code=500, detail="Zig compilation timed out")
            except Exception as e:
                logger.error(f"Zig compilation error: {e}")
                raise HTTPException(status_code=500, detail=f"Compilation error: {e}")

        # Prepare input for Zig client
        zig_request = {
            "model": request.model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
            "stream": request.stream,
            "max_completion_tokens": request.max_completion_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "reasoning_effort": request.reasoning_effort
        }

        if request.stop:
            zig_request["stop"] = request.stop

        # Create temporary file for input
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_input:
            json.dump(zig_request, temp_input)
            temp_input_path = temp_input.name

        try:
            logger.info(f"Calling Zig Cerebras client with model: {request.model}")

            # Call Zig client
            zig_process = subprocess.run(
                [str(zig_binary_path), temp_input_path],
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
                env=dict(os.environ, **{"CEREBRAS_API_KEY": os.getenv("CEREBRAS_API_KEY", "")})
            )

            if zig_process.returncode != 0:
                logger.error(f"Zig client error: {zig_process.stderr}")
                error_msg = zig_process.stderr or "Unknown error from Zig client"

                # Handle specific error cases
                if "MissingApiKey" in error_msg:
                    raise HTTPException(status_code=401, detail="CEREBRAS_API_KEY is missing or invalid")
                elif "InvalidApiKey" in error_msg:
                    raise HTTPException(status_code=401, detail="Invalid Cerebras API key")
                elif "RateLimitExceeded" in error_msg:
                    raise HTTPException(status_code=429, detail="Cerebras API rate limit exceeded")
                elif "ModelNotFound" in error_msg:
                    raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found")
                elif "NetworkError" in error_msg:
                    raise HTTPException(status_code=503, detail="Network error connecting to Cerebras API")
                else:
                    raise HTTPException(status_code=500, detail=f"Cerebras client error: {error_msg}")

            # Parse Zig client output
            try:
                if request.stream:
                    # For streaming responses, return each line as a chunk
                    response_lines = zig_process.stdout.strip().split('\n')
                    full_response = ''.join(response_lines)

                    return {
                        "status": "success",
                        "response": full_response,
                        "model": request.model,
                        "streaming": True,
                        "chunks": len(response_lines),
                        "usage": {
                            "prompt_tokens": sum(len(msg.content.split()) for msg in request.messages),
                            "completion_tokens": len(full_response.split()),
                            "total_tokens": sum(len(msg.content.split()) for msg in request.messages) + len(full_response.split())
                        }
                    }
                else:
                    # Non-streaming response
                    response_data = json.loads(zig_process.stdout)
                    return {
                        "status": "success",
                        "response": response_data.get("content", ""),
                        "model": request.model,
                        "finish_reason": response_data.get("finish_reason", "stop"),
                        "usage": response_data.get("usage", {})
                    }

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Zig client response: {e}")
                logger.error(f"Raw output: {zig_process.stdout}")
                # Return raw output if JSON parsing fails
                return {
                    "status": "success",
                    "response": zig_process.stdout.strip(),
                    "model": request.model,
                    "raw_output": True
                }

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_input_path)
            except:
                pass

    except subprocess.TimeoutExpired:
        logger.error("Zig client timed out")
        raise HTTPException(status_code=504, detail="Request timed out")
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Cerebras endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.post("/api/ai")
async def ai_operations(request: dict):
    """General AI operations endpoint"""
    try:
        operation = request.get("operation", "generate")

        if operation == "generate":
            return await chat_endpoint(ChatRequest(**request))

        elif operation == "vector_search" and "query_text" in request and "documents" in request:
            if unified_app and hasattr(unified_app, 'vector_search_demo'):
                result = unified_app.vector_search_demo(
                    query_text=request["query_text"],
                    documents=request["documents"]
                )
                return {"status": "success", "result": result}
            else:
                # Fallback vector search
                query = request["query_text"]
                docs = request["documents"]
                matches = []
                for i, doc in enumerate(docs):
                    # Simple text similarity based on common words
                    query_words = set(query.lower().split())
                    doc_words = set(doc.lower().split())
                    similarity = len(query_words.intersection(doc_words)) / len(query_words.union(doc_words)) if query_words.union(doc_words) else 0
                    matches.append({
                        "document_id": i,
                        "text": doc[:200] + "..." if len(doc) > 200 else doc,
                        "similarity": round(similarity, 3)
                    })

                # Sort by similarity
                matches.sort(key=lambda x: x["similarity"], reverse=True)

                return {
                    "status": "fallback",
                    "result": {
                        "query": query,
                        "matches": matches[:3],  # Top 3
                        "message": "Vector search running in simulation mode"
                    }
                }

        elif operation == "quantum_demo":
            if unified_app and hasattr(unified_app, 'quantum_computing_demo'):
                result = unified_app.quantum_computing_demo()
                return {"status": "success", "result": result}
            else:
                return {
                    "status": "fallback",
                    "result": {
                        "circuits_created": ["bell_state_simulation", "ghz_state_simulation"],
                        "message": "Quantum computing running in simulation mode"
                    }
                }

        else:
            return {"status": "error", "message": f"Unknown operation: {operation}"}

    except Exception as e:
        logger.error(f"AI operations error: {e}")
        raise HTTPException(status_code=500, detail=f"AI operations error: {e}")

# WebSocket endpoint for real-time communication (optional)
@app.get("/api/status")
async def system_status():
    """Get comprehensive system status"""
    return {
        "backend": "FastAPI",
        "version": "1.0.0",
        "status": "active",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "quantum_integration": QUANTUM_AVAILABLE,
            "unified_app": UNIFIED_APP_AVAILABLE,
            "integrations": INTEGRATIONS_AVAILABLE,
            "endpoints": ["/api/chat", "/api/services", "/api/quantum", "/api/ai"]
        },
        "environment": {
            "port": 5000,
            "host": "0.0.0.0",
            "cors_enabled": True
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": f"Path {request.url.path} not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

if __name__ == "__main__":
    # Check for environment variables and log status
    logger.info("🔧 Environment Configuration:")
    logger.info(f"   NVIDIA_API_KEY: {'✅ Set' if os.getenv('NVIDIA_API_KEY') else '❌ Not set'}")
    logger.info(f"   IBM_QUANTUM_API_TOKEN: {'✅ Set' if os.getenv('IBM_QUANTUM_API_TOKEN') else '❌ Not set'}")
    logger.info(f"   UPSTASH_REDIS_REST_URL: {'✅ Set' if os.getenv('UPSTASH_REDIS_REST_URL') else '❌ Not set'}")

    logger.info("🚀 Starting FastAPI server on 0.0.0.0:5000")

    # Run the server
    uvicorn.run(
        "fastapi_backend:app",
        host="0.0.0.0",
        port=5000,
        reload=False,
        log_level="info",
        access_log=True
    )
# =======================================================================


# =======================================================================
