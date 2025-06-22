import os
import PyPDF2
import faiss
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import json
import tiktoken
from openai import OpenAI
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StatementAnalyst:
    """
    OpenAI Agent SDK-based Statement Analyst using text-embedding-3-small and GPT-4.1 nano
    """
    
    def __init__(self):
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Model configurations
        self.embedding_model = "text-embedding-3-small"
        self.chat_model = "gpt-4.1-mini-2025-04-14"  # GPT-4.1 mini for enhanced quality
        self.embedding_dimension = 1536  # text-embedding-3-small dimension
        
        # Token counter for cost optimization
        # Using o200k_base encoding for GPT-4.1 mini (fallback to compatible encoding)
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4.1-mini-2025-04-14")
        except KeyError:
            # Fallback to o200k_base encoding which is compatible with GPT-4 models
            self.encoding = tiktoken.get_encoding("o200k_base")
        
        # Chunk configuration optimized for financial documents
        self.chunk_size = 800  # Optimal for financial statements
        self.chunk_overlap = 100
        
        # Storage for embeddings and metadata
        self.statements_db = {}
        
        # Create storage directories
        os.makedirs("storage/statements", exist_ok=True)
        
        # Load existing statements on startup
        self._load_existing_statements()
        
        # System prompt for the agent
        self.system_prompt = """You are a specialized Financial Statement Analysis Agent. Your expertise includes:

1. **Bank Statement Analysis**: Extract and interpret account balances, transactions, fees, and financial patterns
2. **Investment Portfolio Review**: Analyze holdings, asset allocation, performance metrics
3. **Transaction Analysis**: Categorize expenses, identify trends, flag unusual activities
4. **Financial Health Assessment**: Evaluate cash flow, spending patterns, and account activity

**Instructions:**
- Base ALL responses strictly on the provided statement content
- If information isn't in the statement, clearly state "This information is not available in the provided statement"
- Provide specific page/section references when possible
- Use exact figures and dates from the statement
- Explain financial terms when relevant
- Be concise but comprehensive in your analysis

**Response Format:**
- Direct answers with supporting evidence from the statement
- Include relevant context and implications
- Suggest follow-up questions when helpful"""

    async def process_statement(self, file_path: str, session_id: str, filename: str) -> Dict:
        """
        Process uploaded PDF statement using OpenAI's embedding API
        """
        try:
            logger.info(f"Processing statement {filename} for session {session_id}")
            
            # Extract text from PDF with page tracking
            pages_content = self._extract_pdf_text_with_pages(file_path)
            
            if not pages_content:
                raise ValueError("No text content found in PDF")
            
            # Create chunks with page context
            chunks = self._create_intelligent_chunks(pages_content)
            logger.info(f"Created {len(chunks)} intelligent chunks")
            
            # Generate embeddings using OpenAI API
            embeddings = await self._generate_openai_embeddings(chunks)
            
            # Create FAISS index
            index = faiss.IndexFlatL2(self.embedding_dimension)
            embeddings_array = np.array(embeddings).astype('float32')
            index.add(embeddings_array)
            
            # Store everything for this session
            statement_data = {
                "filename": filename,
                "chunks": chunks,
                "embeddings": embeddings,
                "faiss_index": index,
                "processed_at": datetime.now().isoformat(),
                "chunk_count": len(chunks),
                "total_pages": len(pages_content),
                "model_used": {
                    "embedding": self.embedding_model,
                    "chat": self.chat_model
                }
            }
            
            self.statements_db[session_id] = statement_data
            
            # Save to disk for persistence
            await self._save_statement_data(session_id, statement_data)
            
            return {
                "status": "success",
                "message": f"Statement '{filename}' processed successfully with OpenAI Agent SDK",
                "chunk_count": len(chunks),
                "pages_processed": len(pages_content),
                "ready_for_queries": True
            }
            
        except Exception as e:
            logger.error(f"Error processing statement: {str(e)}")
            return {
                "status": "error", 
                "message": str(e),
                "ready_for_queries": False
            }

    async def query_statement(self, query: str, session_id: str, top_k: int = 5) -> Dict:
        """
        Query statement using OpenAI Agent SDK with GPT-4.1 nano
        """
        try:
            # Check if statement exists
            if session_id not in self.statements_db:
                return {
                    "status": "error",
                    "message": "No statement found. Please upload a statement first.",
                    "answer": None
                }
            
            statement_data = self.statements_db[session_id]
            
            # Generate query embedding using OpenAI
            query_embedding = await self._generate_openai_embeddings([query])
            
            # Search for most relevant chunks
            distances, indices = statement_data["faiss_index"].search(
                np.array(query_embedding).astype('float32'), top_k
            )
            
            # Get relevant chunks with metadata
            relevant_chunks = []
            for i, idx in enumerate(indices[0]):
                if idx < len(statement_data["chunks"]):
                    chunk_data = statement_data["chunks"][idx]
                    relevant_chunks.append({
                        "content": chunk_data["content"],
                        "page": chunk_data.get("page", "Unknown"),
                        "section": chunk_data.get("section", ""),
                        "similarity_score": float(1 - distances[0][i])  # Convert distance to similarity
                    })
            
            # Generate answer using GPT-4.1 nano
            answer = await self._generate_agent_response(query, relevant_chunks, statement_data["filename"])
            
            return {
                "status": "success",
                "answer": answer,
                "relevant_chunks": len(relevant_chunks),
                "source_document": statement_data["filename"],
                "model_used": self.chat_model
            }
            
        except Exception as e:
            logger.error(f"Error querying statement: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "answer": None
            }

    def _extract_pdf_text_with_pages(self, file_path: str) -> List[Dict]:
        """
        Extract text content from PDF with page tracking and section identification
        """
        pages_content = []
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            # Identify sections (basic pattern matching for financial statements)
                            section = self._identify_section(page_text)
                            
                            pages_content.append({
                                "page_number": page_num + 1,
                                "content": page_text.strip(),
                                "section": section,
                                "word_count": len(page_text.split())
                            })
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading PDF file: {e}")
            raise
        
        return pages_content

    def _identify_section(self, text: str) -> str:
        """
        Identify the section type based on content patterns
        """
        text_lower = text.lower()
        
        # Common financial statement sections
        if any(word in text_lower for word in ['account summary', 'summary', 'overview']):
            return "Account Summary"
        elif any(word in text_lower for word in ['transaction', 'activity', 'movement']):
            return "Transaction History"
        elif any(word in text_lower for word in ['balance', 'positions', 'holdings']):
            return "Account Balances"
        elif any(word in text_lower for word in ['fee', 'charge', 'cost']):
            return "Fees and Charges"
        elif any(word in text_lower for word in ['investment', 'portfolio', 'securities']):
            return "Investment Details"
        else:
            return "General"

    def _create_intelligent_chunks(self, pages_content: List[Dict]) -> List[Dict]:
        """
        Create intelligent chunks optimized for financial document structure
        """
        chunks = []
        
        for page_data in pages_content:
            text = page_data["content"]
            page_num = page_data["page_number"]
            section = page_data["section"]
            
            # Split text into sentences for better semantic boundaries
            sentences = text.replace('\n', ' ').split('. ')
            
            current_chunk = ""
            current_token_count = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Add period back if it was removed
                if not sentence.endswith('.') and not sentence.endswith('?') and not sentence.endswith('!'):
                    sentence += '.'
                
                # Count tokens
                sentence_tokens = len(self.encoding.encode(sentence))
                
                # Check if adding this sentence would exceed chunk size
                if current_token_count + sentence_tokens > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append({
                        "content": current_chunk.strip(),
                        "page": page_num,
                        "section": section,
                        "token_count": current_token_count
                    })
                    
                    # Start new chunk with overlap
                    if len(chunks) > 0:
                        # Include last few sentences for context
                        overlap_sentences = current_chunk.split('. ')[-2:]
                        current_chunk = '. '.join(overlap_sentences) + '. ' + sentence
                        current_token_count = len(self.encoding.encode(current_chunk))
                    else:
                        current_chunk = sentence
                        current_token_count = sentence_tokens
                else:
                    current_chunk += ' ' + sentence
                    current_token_count += sentence_tokens
            
            # Add final chunk if it has content
            if current_chunk.strip():
                chunks.append({
                    "content": current_chunk.strip(),
                    "page": page_num,
                    "section": section,
                    "token_count": current_token_count
                })
        
        return chunks

    async def _generate_openai_embeddings(self, chunks: List[Any]) -> List[List[float]]:
        """
        Generate embeddings using OpenAI's text-embedding-3-small
        """
        try:
            # Prepare text input
            if isinstance(chunks[0], dict):
                texts = [chunk["content"] for chunk in chunks]
            else:
                texts = chunks
            
            # Generate embeddings in batches to avoid rate limits
            batch_size = 50
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch_texts,
                    encoding_format="float"
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    async def _generate_agent_response(self, query: str, relevant_chunks: List[Dict], filename: str) -> str:
        """
        Generate response using GPT-4.1 nano with the agent system prompt
        """
        try:
            # Prepare context from relevant chunks
            context_parts = []
            for i, chunk in enumerate(relevant_chunks):
                context_parts.append(
                    f"**Source {i+1} (Page {chunk['page']}, {chunk['section']}):**\n{chunk['content']}\n"
                )
            
            context = "\n".join(context_parts)
            
            # Create the user message with context
            user_message = f"""**Document:** {filename}

**User Question:** {query}

**Relevant Content from Statement:**
{context}

Please analyze the above content and provide a comprehensive answer to the user's question. Reference specific details from the statement and include page numbers when relevant."""

            # Generate response using GPT-4.1 nano
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,  # Low temperature for factual accuracy
                max_tokens=1000,
                top_p=0.9
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating agent response: {e}")
            return f"I apologize, but I encountered an error while analyzing your statement. Please try rephrasing your question or contact support if the issue persists."

    async def _save_statement_data(self, session_id: str, statement_data: Dict):
        """
        Save statement data to disk for persistence
        """
        try:
            # Save metadata and chunks (excluding FAISS index)
            save_data = {
                "filename": statement_data["filename"],
                "chunks": statement_data["chunks"],
                "processed_at": statement_data["processed_at"],
                "chunk_count": statement_data["chunk_count"],
                "total_pages": statement_data["total_pages"],
                "model_used": statement_data["model_used"]
            }
            
            with open(f"storage/statements/{session_id}.json", 'w') as f:
                json.dump(save_data, f, indent=2)
            
            # Save FAISS index
            faiss.write_index(statement_data["faiss_index"], f"storage/statements/{session_id}.faiss")
            
            # Save embeddings
            np.save(f"storage/statements/{session_id}_embeddings.npy", statement_data["embeddings"])
            
            logger.info(f"Saved statement data for session {session_id}")
            
        except Exception as e:
            logger.warning(f"Could not save statement data: {e}")

    def load_statement_data(self, session_id: str) -> bool:
        """
        Load previously processed statement data
        """
        try:
            # Load metadata
            with open(f"storage/statements/{session_id}.json", 'r') as f:
                save_data = json.load(f)
            
            # Load FAISS index
            index = faiss.read_index(f"storage/statements/{session_id}.faiss")
            
            # Load embeddings
            embeddings = np.load(f"storage/statements/{session_id}_embeddings.npy").tolist()
            
            # Reconstruct statement data
            statement_data = {
                **save_data,
                "faiss_index": index,
                "embeddings": embeddings
            }
            
            self.statements_db[session_id] = statement_data
            logger.info(f"Loaded statement data for session {session_id}")
            return True
            
        except Exception as e:
            logger.info(f"Could not load statement data for session {session_id}: {e}")
            return False

    def has_statement(self, session_id: str) -> bool:
        """Check if a statement is loaded for this session"""
        return session_id in self.statements_db

    def get_statement_info(self, session_id: str) -> Optional[Dict]:
        """Get basic info about uploaded statement"""
        if session_id not in self.statements_db:
            return None
        
        data = self.statements_db[session_id]
        return {
            "filename": data["filename"],
            "processed_at": data["processed_at"],
            "chunk_count": data["chunk_count"],
            "total_pages": data["total_pages"],
            "model_used": data["model_used"],
            "ready_for_queries": True
        }

    def _load_existing_statements(self):
        """Load all previously processed statements on startup"""
        try:
            storage_dir = Path("storage/statements")
            if not storage_dir.exists():
                return
            
            # Find all JSON files (statement metadata)
            json_files = list(storage_dir.glob("*.json"))
            loaded_count = 0
            
            for json_file in json_files:
                session_id = json_file.stem  # Filename without extension
                if self.load_statement_data(session_id):
                    loaded_count += 1
                    logger.info(f"Loaded existing statement for session: {session_id}")
            
            if loaded_count > 0:
                logger.info(f"Loaded {loaded_count} existing statements on startup")
            else:
                logger.info("No existing statements found")
                
        except Exception as e:
            logger.error(f"Error loading existing statements: {e}")

# Test function
async def test_openai_agent():
    """Test the OpenAI Agent SDK implementation"""
    analyst = StatementAnalyst()
    
    # Test with sample chunks
    test_chunks = [
        {
            "content": "Account Summary as of December 31, 2024: Total Balance $125,450.00. Checking Account: $15,450.00, Savings Account: $35,000.00, Investment Account: $75,000.00",
            "page": 1,
            "section": "Account Summary"
        },
        {
            "content": "Investment Holdings: Apple Inc. (AAPL) 100 shares at $150.00 = $15,000.00, Microsoft Corp (MSFT) 80 shares at $250.00 = $20,000.00, Cash Position: $40,000.00",
            "page": 2,
            "section": "Investment Details"
        }
    ]
    
    # Generate embeddings
    embeddings = await analyst._generate_openai_embeddings(test_chunks)
    
    # Create FAISS index
    index = faiss.IndexFlatL2(analyst.embedding_dimension)
    embeddings_array = np.array(embeddings).astype('float32')
    index.add(embeddings_array)
    
    # Store test data
    analyst.statements_db["test"] = {
        "filename": "test_statement.pdf",
        "chunks": test_chunks,
        "faiss_index": index,
        "embeddings": embeddings,
        "processed_at": datetime.now().isoformat(),
        "chunk_count": len(test_chunks),
        "total_pages": 2,
        "model_used": {
            "embedding": analyst.embedding_model,
            "chat": analyst.chat_model
        }
    }
    
    # Test query
    result = await analyst.query_statement("What is my total account balance?", "test")
    print("OpenAI Agent Test Result:", result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_openai_agent())
