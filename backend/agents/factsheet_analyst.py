import os
import json
import logging
import requests
import aiohttp
from typing import Dict, List, Optional
from datetime import datetime
from openai import OpenAI
import vectorize_client as v

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FactsheetAnalyst:
    """
    Vectorize.io-based Factsheet Analyst Agent
    Integrates with pre-uploaded factsheets on Vectorize.io
    """
    
    def __init__(self):
        self.client = OpenAI()
        self.chat_model = "gpt-4.1-mini-2025-04-14"
        
        # Vectorize.io configuration
        self.vectorize_api_token = os.getenv("VECTORIZE_API_TOKEN")  # Changed from VECTORIZE_API_KEY
        self.vectorize_org_id = os.getenv("VECTORIZE_ORG_ID")  # Organization ID
        self.vectorize_pipeline_id = os.getenv("VECTORIZE_PIPELINE_ID")  # Pipeline ID for factsheets
        self.vectorize_base_url = "https://api.vectorize.io/v1"
        
        if not self.vectorize_api_token:
            logger.warning("VECTORIZE_API_TOKEN not found in environment variables")
        if not self.vectorize_org_id:
            logger.warning("VECTORIZE_ORG_ID not found in environment variables")
        if not self.vectorize_pipeline_id:
            logger.warning("VECTORIZE_PIPELINE_ID not found in environment variables")
        
        # Flag to track initialization
        self._initialized = False
        
        # Available factsheets metadata (loaded from Vectorize.io)
        self.available_factsheets = []
        
        # Initialize Vectorize API client
        self.vectorize_api_client = None
        self.pipelines_api = None
        if self._check_vectorize_config():
            try:
                api_config = v.Configuration(access_token=self.vectorize_api_token)
                self.vectorize_api_client = v.ApiClient(api_config)
                self.pipelines_api = v.PipelinesApi(self.vectorize_api_client)
                logger.info("Vectorize API client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Vectorize API client: {e}")
        
        # System prompt for factsheet analysis
        self.system_prompt = """You are a specialized Investment Factsheet Analyst with access to a comprehensive database of investment factsheets. Your expertise includes:

1. **Factsheet Analysis**: Extract key information from investment factsheets, fund documents, and prospectuses
2. **Performance Analysis**: Interpret returns, fees, risk metrics, and benchmarks
3. **Investment Strategy**: Understand and explain investment objectives, asset allocation, and methodology
4. **Risk Assessment**: Analyze risk factors, volatility metrics, and regulatory disclosures
5. **Comparative Analysis**: Compare multiple funds and investment options

**Instructions:**
- Base ALL responses on the provided factsheet content from Vectorize.io
- Reference specific factsheet names and sections when available
- Explain financial metrics and terminology clearly
- Highlight important risks and considerations
- Compare multiple factsheets when relevant
- Always cite the source factsheet name
- Provide actionable insights for investment decisions

**Response Format:**
- Clear, structured answers with supporting data
- Include relevant performance metrics and comparisons
- Highlight key investment considerations and risks
- Suggest follow-up questions when helpful
- Reference specific factsheet documents used"""

    async def initialize(self):
        """
        Initialize the Vectorize.io connection and load available factsheets
        """
        if not self._initialized:
            await self._load_available_factsheets()
            self._initialized = True
            logger.info("VectorizeIO FactsheetAnalyst initialized successfully")

    async def _load_available_factsheets(self):
        """
        Load metadata about available factsheets from Vectorize.io
        """
        try:
            if not self._check_vectorize_config():
                logger.warning("Vectorize.io credentials not configured")
                return
            
            if not self.pipelines_api:
                logger.warning("Vectorize API client not initialized")
                return
            
            # Test connection by performing a simple retrieval
            try:
                test_request = v.RetrieveDocumentsRequest(
                    question="test connection",
                    num_results=1
                )
                response = self.pipelines_api.retrieve_documents(
                    self.vectorize_org_id, 
                    self.vectorize_pipeline_id, 
                    test_request
                )
                
                # Count available documents
                doc_count = len(response.documents) if hasattr(response, 'documents') else 0
                self.available_factsheets = [{"name": f"Document {i+1}"} for i in range(min(doc_count, 10))]  # Placeholder
                
                logger.info(f"Successfully connected to Vectorize.io pipeline with {doc_count} documents")
                
            except Exception as e:
                logger.warning(f"Error testing Vectorize.io connection: {e}")
                        
        except Exception as e:
            logger.error(f"Error loading factsheets from Vectorize.io: {e}")

    def _check_vectorize_config(self) -> bool:
        """Check if Vectorize.io is properly configured"""
        return bool(
            self.vectorize_api_token and 
            self.vectorize_org_id and 
            self.vectorize_pipeline_id
        )

    async def search_factsheets(self, query: str, session_id: str, top_k: int = 5) -> Dict:
        """
        Search factsheets using Vectorize.io
        """
        try:
            if not self._check_vectorize_config():
                return {
                    "status": "error",
                                    "message": "Vectorize.io not configured. Please check API credentials.",
                "answer": "I don't have access to factsheet data. Please configure Vectorize.io integration with VECTORIZE_API_TOKEN, VECTORIZE_ORG_ID, and VECTORIZE_PIPELINE_ID environment variables."
                }
            
            # Search using Vectorize.io API
            search_results = await self._vectorize_search(query, top_k)
            
            if not search_results:
                return {
                    "status": "error",
                    "message": "Unable to access factsheet data.",
                    "answer": "I can't access the factsheet data right now. This may be due to a plan limitation with Vectorize.io. Please check your Vectorize.io plan or try accessing factsheets directly through the Vectorize.io dashboard."
                }
            
            # Generate answer using the search results
            answer = await self._generate_factsheet_answer(query, search_results)
            
            return {
                "status": "success",
                "answer": answer,
                "sources": [result.get("metadata", {}).get("source", "Unknown") for result in search_results],
                "results_count": len(search_results)
            }
            
        except Exception as e:
            logger.error(f"Error searching factsheets: {e}")
            return {
                "status": "error",
                "message": str(e),
                "answer": "I encountered an error while searching the factsheets. Please try again."
            }

    async def _vectorize_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform semantic search using Vectorize.io official client
        """
        try:
            if not self.pipelines_api:
                logger.error("Vectorize API client not initialized")
                return []
            
            # Use the official Vectorize client for retrieval (available on free plan)
            request = v.RetrieveDocumentsRequest(
                question=query,
                num_results=top_k
            )
            
            # Call the retrieval endpoint using the official client
            response = self.pipelines_api.retrieve_documents(
                self.vectorize_org_id, 
                self.vectorize_pipeline_id, 
                request
            )
            
            documents = response.documents if hasattr(response, 'documents') else []
            
            # Transform Vectorize.io response format to our expected format
            transformed_results = []
            for doc in documents:
                transformed_results.append({
                    "metadata": {
                        "text": doc.content if hasattr(doc, 'content') else str(doc),
                        "source": getattr(doc, 'metadata', {}).get("source", "Unknown Document"),
                        "page": getattr(doc, 'metadata', {}).get("page", ""),
                    },
                    "score": getattr(doc, 'relevance_score', 0.0)
                })
            
            logger.info(f"Found {len(transformed_results)} factsheet documents using official client")
            return transformed_results
                        
        except Exception as e:
            logger.error(f"Error in Vectorize.io search using official client: {e}")
            return []

    async def _generate_factsheet_answer(self, query: str, search_results: List[Dict]) -> str:
        """
        Generate comprehensive answer using GPT-4o mini with factsheet data
        """
        try:
            # Prepare context from search results
            context_parts = []
            sources = set()
            
            for i, result in enumerate(search_results):
                metadata = result.get("metadata", {})
                content = metadata.get("text", "")
                source = metadata.get("source", f"Document {i+1}")
                page = metadata.get("page", "")
                score = result.get("score", 0)
                
                if content:
                    sources.add(source)
                    context_parts.append(
                        f"**Source: {source}** (Relevance: {score:.3f})\n"
                        f"Content: {content}\n"
                    )
            
            if not context_parts:
                return "I found some factsheets but couldn't extract readable content. Please try rephrasing your question."
            
            context = "\n".join(context_parts)
            
            # Create the prompt for GPT-4o mini
            user_message = f"""**Investment Query:** {query}

**Factsheet Information from Vectorize.io:**
{context}

**Sources Referenced:** {', '.join(sources)}

Please provide a comprehensive analysis based on the factsheet information above. Include specific details, metrics, and insights from the documents. Cite the source documents in your response."""

            # Generate response using GPT-4o mini
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,
                max_tokens=1000,
                top_p=0.9
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating factsheet answer: {e}")
            return f"I found relevant factsheet information but encountered an error generating the response. Error: {str(e)}"

    # Legacy methods for backward compatibility
    async def ingest_factsheet(self, file_path: str, filename: str, metadata: Dict = None) -> Dict:
        """
        Placeholder for factsheet ingestion - now handled by Vectorize.io
        """
        return {
            "status": "info",
            "message": f"Factsheets are now managed through Vectorize.io. The file '{filename}' should be uploaded directly to your Vectorize.io index.",
            "filename": filename
        }

    def get_available_factsheets(self) -> Dict:
        """
        Return list of available factsheets from Vectorize.io
        """
        if not self._check_vectorize_config():
            return {
                "factsheets": [],
                "status": "configuration_required",
                "message": "Vectorize.io credentials not properly configured. Please check your environment variables.",
                "required_vars": ["VECTORIZE_API_TOKEN", "VECTORIZE_ORG_ID", "VECTORIZE_PIPELINE_ID"]
            }
        
        # Check if API client is properly initialized
        if not self.pipelines_api:
            return {
                "factsheets": [],
                "status": "error",
                "message": "Vectorize.io API client not initialized. Please restart the application."
            }
        
        # Return success status since we know the connection works
        return {
            "factsheets": self.available_factsheets if self.available_factsheets else [{"name": "Vectorize.io Pipeline Connected"}],
            "status": "success",
            "message": f"Connected to Vectorize.io pipeline. {len(self.available_factsheets) if self.available_factsheets else 'Documents'} available for querying."
        }

    # Methods for orchestrator compatibility
    async def analyze_fund_performance(self, fund_name: str, session_id: str) -> Dict:
        """Analyze specific fund performance"""
        query = f"Performance analysis and metrics for {fund_name} fund including returns, volatility, and benchmarks"
        return await self.search_factsheets(query, session_id)

    async def compare_investments(self, fund_names: List[str], session_id: str) -> Dict:
        """Compare multiple investment options"""
        funds_list = ", ".join(fund_names)
        query = f"Compare investment funds: {funds_list}. Include fees, performance, risk levels, and investment strategies"
        return await self.search_factsheets(query, session_id)

# Backward compatibility
class FactsheetWatcher:
    """Placeholder for file system watcher - not needed with Vectorize.io"""
    def __init__(self, factsheet_analyst):
        logger.info("FactsheetWatcher: Using Vectorize.io integration, file watching disabled")

def start_factsheet_watcher(factsheet_analyst):
    """Placeholder function for backward compatibility"""
    logger.info("Factsheet watching: Using Vectorize.io integration")
    return None
