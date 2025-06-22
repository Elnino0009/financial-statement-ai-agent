from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import shutil
import json
from pathlib import Path
import uuid
import asyncio
import logging
from typing import Dict, List
from dotenv import load_dotenv

from agents.orchestrator import FinancialOrchestrator
from agents.statement_analyst import StatementAnalyst
from agents.factsheet_analyst import FactsheetAnalyst
from agents.news_agent import NewsAgent

# Load environment variables
load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize agents (without async operations)
statement_analyst = StatementAnalyst()
factsheet_analyst = FactsheetAnalyst()
news_agent = NewsAgent()

# This will be initialized in lifespan event
orchestrator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI startup and shutdown
    
    Why use lifespan instead of on_event:
    1. Modern FastAPI recommended approach
    2. Ensures event loop is running
    3. Proper async initialization
    4. Error handling during startup
    5. Clean separation of sync/async initialization
    """
    global orchestrator
    
    # Startup
    try:
        # Initialize factsheet analyst (this will load existing factsheets)
        await factsheet_analyst.initialize()
        
        # Now initialize orchestrator with properly initialized agents
        orchestrator = FinancialOrchestrator()
        orchestrator.agents['statement_analyst'] = statement_analyst
        orchestrator.agents['factsheet_analyst'] = factsheet_analyst
        orchestrator.agents['news_agent'] = news_agent
        
        print("✅ All agents initialized successfully")
        
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        raise
    
    yield
    
    # Shutdown (cleanup if needed)
    print("🔄 Shutting down Financial AI Platform")

app = FastAPI(
    title="Financial AI Platform - Multi-Agent System",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
FACTSHEETS_DIR = Path("factsheets")
FACTSHEETS_DIR.mkdir(exist_ok=True)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        print(f"WebSocket connected for session: {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            print(f"WebSocket disconnected for session: {session_id}")
    
    async def send_update(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(json.dumps(message))
            except Exception as e:
                print(f"Error sending WebSocket message: {e}")
                self.disconnect(session_id)

manager = ConnectionManager()

@app.get("/")
async def root():
    return {
        "message": "Financial AI Platform - Multi-Agent System Ready!",
        "agents": ["orchestrator", "statement_analyst", "factsheet_analyst", "news_agent"],
        "models": {
            "embedding": "text-embedding-3-small",
                            "chat": "gpt-4.1-mini-2025-04-14"
        },
        "initialized": orchestrator is not None
    }

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    
    # Send immediate welcome message to confirm connection
    await manager.send_update(session_id, {
        "type": "connection_established",
        "message": "WebSocket connection established",
        "session_id": session_id,
        "timestamp": str(asyncio.get_event_loop().time())
    })
    
    try:
        # Keep connection alive and handle any incoming messages
        while True:
            try:
                # Wait for messages from client (optional)
                data = await websocket.receive_text()
                # Echo back for testing
                await manager.send_update(session_id, {
                    "type": "message_received",
                    "message": f"Received: {data}",
                    "timestamp": str(asyncio.get_event_loop().time())
                })
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(session_id)

@app.post("/api/upload-statement")
async def upload_statement(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files supported")
        
        session_id = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{session_id}_{file.filename}"
        
        await manager.send_update(session_id, {
            "type": "upload_started",
            "filename": file.filename,
            "progress": 0
        })
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        await manager.send_update(session_id, {
            "type": "upload_progress",
            "progress": 50,
            "message": "File uploaded, starting analysis..."
        })
        
        result = await statement_analyst.process_statement(
            str(file_path), session_id, file.filename
        )
        
        file_path.unlink()
        
        if result["status"] == "success":
            await manager.send_update(session_id, {
                "type": "upload_completed",
                "progress": 100,
                "message": "Statement processed successfully"
            })
            
            return {
                "session_id": session_id,
                "filename": file.filename,
                "status": "success",
                "message": result["message"],
                "chunk_count": result["chunk_count"]
            }
        else:
            raise HTTPException(status_code=500, detail=result["message"])
            
    except Exception as e:
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-factsheet")
async def upload_factsheet(file: UploadFile = File(...)):
    """
    Factsheet upload endpoint - now redirects to Vectorize.io information
    """
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF factsheets supported")
        
        # Since factsheets are now managed via Vectorize.io, we return information
        # instead of processing the file
        return {
            "filename": file.filename,
            "status": "info",
            "message": "Factsheets are now managed through Vectorize.io for enhanced performance and scalability.",
            "details": {
                "integration": "Vectorize.io",
                "capabilities": [
                    "Lightning-fast semantic search",
                    "Pre-indexed investment documents",
                    "Real-time query processing",
                    "Scalable vector database"
                ],
                "next_steps": "Ask questions about investment factsheets directly in the chat. The AI will search across all available documents automatically."
            },
            "available_queries": [
                "Compare fund performance metrics",
                "Analyze investment fees and expenses", 
                "Find ESG investment options",
                "Show risk ratings across funds"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in factsheet upload endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/orchestrated-chat")
async def orchestrated_chat(query_data: dict):
    try:
        if orchestrator is None:
            raise HTTPException(status_code=503, detail="Orchestrator not initialized")
        
        query = query_data.get("query", "")
        session_id = query_data.get("session_id", "")
        
        if not query:
            raise HTTPException(status_code=400, detail="Query required")
        
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID required")
        
        # Check if statement exists and log the result
        has_statement = statement_analyst.has_statement(session_id)
        available_factsheets = factsheet_analyst.get_available_factsheets()
        
        logger.info(f"Orchestrated chat - Session: {session_id}, Has statement: {has_statement}, Available factsheets: {len(available_factsheets)}")
        
        await manager.send_update(session_id, {
            "type": "orchestration_started",
            "query": query,
            "status": "planning"
        })
        
        context = {
            "has_statement": has_statement,
            "available_factsheets": available_factsheets
        }
        
        logger.info(f"Context being passed to orchestrator: {context}")
        
        result = await orchestrator.process_query(query, session_id, context)
        
        await manager.send_update(session_id, {
            "type": "orchestration_completed",
            "status": "completed",
            "execution_id": result.get("execution_id")
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Error in orchestrated chat: {str(e)}", exc_info=True)
        await manager.send_update(session_id, {
            "type": "orchestration_error",
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/factsheets")
async def get_available_factsheets():
    return factsheet_analyst.get_available_factsheets()

@app.get("/api/session/{session_id}/status")
async def get_session_status(session_id: str):
    return {
        "session_id": session_id,
        "has_statement": statement_analyst.has_statement(session_id),
        "statement_info": statement_analyst.get_statement_info(session_id),
        "available_factsheets": len(factsheet_analyst.get_available_factsheets()),
        "websocket_connected": session_id in manager.active_connections
    }

@app.get("/api/debug/sessions")
async def debug_sessions():
    """Debug endpoint to see all active sessions"""
    return {
        "active_sessions": list(statement_analyst.statements_db.keys()),
        "session_details": {
            session_id: {
                "filename": data.get("filename"),
                "chunk_count": data.get("chunk_count"),
                "processed_at": data.get("processed_at")
            }
            for session_id, data in statement_analyst.statements_db.items()
        }
    }

@app.post("/api/chat")
async def direct_chat(query_data: dict):
    """
    Direct chat endpoint for simple statement queries
    This handles the ChatInterface component that calls /api/chat
    """
    try:
        query = query_data.get("query", "")
        session_id = query_data.get("session_id", "")
        
        if not query:
            raise HTTPException(status_code=400, detail="Query required")
        
        if not session_id:
            return {
                "response": "It seems that I currently don't have access to your financial statements. Please upload a statement first to get started.",
                "source": "system",
                "document": None
            }
        
        # Check if statement exists for this session
        if not statement_analyst.has_statement(session_id):
            return {
                "response": "It seems that I currently don't have access to your financial statements. Please upload a statement first, then I'll be able to answer questions about your financial data.",
                "source": "system", 
                "document": None
            }
        
        # Query the statement directly
        result = await statement_analyst.query_statement(query, session_id)
        
        if result["status"] == "success":
            return {
                "response": result["answer"],
                "source": "statement_analyst",
                "document": result["source_document"]
            }
        else:
            return {
                "response": f"I encountered an issue: {result['message']}. Please try rephrasing your question.",
                "source": "error",
                "document": None
            }
            
    except Exception as e:
        logger.error(f"Error in direct chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
