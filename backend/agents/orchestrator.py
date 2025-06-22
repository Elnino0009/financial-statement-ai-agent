import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from openai import OpenAI
from dataclasses import dataclass
from enum import Enum
from .news_agent import NewsAgent

from .statement_analyst import StatementAnalyst
from .factsheet_analyst import FactsheetAnalyst
from .news_agent import NewsAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class AgentExecution:
    agent_name: str
    status: AgentStatus
    progress: float
    message: str
    result: Optional[Dict] = None
    error: Optional[str] = None

class FinancialOrchestrator:
    """
    OpenAI Agent SDK-based Orchestrator that coordinates multiple financial AI agents
    """
    
    def __init__(self):
        self.client = OpenAI()
        self.model = "gpt-4.1-mini-2025-04-14"  # GPT-4.1 mini for enhanced quality
        
        # Initialize specialized agents
        self.agents = {}
        
        # Execution tracking
        self.active_executions = {}  # session_id -> execution status
        
        # System prompt for orchestration planning
        self.orchestration_prompt = """You are a Financial AI Orchestrator using GPT-4.1 nano for optimal performance and cost efficiency. Your role is to analyze user queries and create intelligent execution plans.

        Available Agents:
        1. **statement_analyst**: Analyzes uploaded bank statements using RAG with text-embedding-3-small
        - Actions: query_statement, get_positions, extract_transactions
        
        2. **factsheet_analyst**: Searches investment factsheets using vector embeddings
        - Actions: search_factsheets, analyze_fund_performance, compare_investments
        
        3. **news_agent**: Fetches relevant financial news using multiple APIs
        - Actions: get_asset_news, get_market_news
        - Sources: NewsAPI.org, Alpha Vantage (free tiers)

        **Planning Rules:**
        - Use statement_analyst for: portfolio analysis, position queries, transaction history
        - Use factsheet_analyst for: investment research, fund analysis, document searches  
        - Use news_agent for: market updates, asset-specific news, sentiment analysis
        - Combine multiple agents for comprehensive analysis

        **CRITICAL: Always include the user's query in the inputs when using statement_analyst or factsheet_analyst**

        **Response Format:**
        You must respond with valid JSON in the following format:
        {
            "intent": "user_intent_classification",
            "agents_needed": ["agent1", "agent2"],
            "execution_sequence": [
                {
                    "agent": "agent_name", 
                    "action": "specific_action",
                    "inputs": {"query": "user's original question", "assets": ["AAPL", "MSFT"], "days_back": 7},
                    "dependencies": []
                }
            ],
            "expected_output": "description_of_expected_result"
        }

        **Examples:**
        For "What's my account balance?":
        {
            "intent": "account_balance_query",
            "agents_needed": ["statement_analyst"],
            "execution_sequence": [
                {
                    "agent": "statement_analyst",
                    "action": "query_statement", 
                    "inputs": {"query": "What's my account balance?"}
                }
            ]
        }

        Optimize for the GPT-4.1 nano model's strengths in instruction following and efficiency."""


    async def process_query(self, user_query: str, session_id: str, context: Dict = None) -> Dict:
        """
        Main orchestration method that processes user queries using multiple agents
        """
        try:
            logger.info(f"Processing query for session {session_id}: {user_query}")
            
            # Initialize execution tracking
            execution_id = f"{session_id}_{datetime.now().timestamp()}"
            self.active_executions[execution_id] = {
                "session_id": session_id,
                "query": user_query,
                "status": AgentStatus.PLANNING,
                "agents": [],
                "start_time": datetime.now(),
                "context": context or {}
            }
            
            # Step 1: Create execution plan
            execution_plan = await self._create_execution_plan(user_query, context)
            
            if not execution_plan or "agents_needed" not in execution_plan:
                return {
                    "response": "I couldn't determine how to help with that query. Please try asking about your financial statements, investments, or market news.",
                    "execution_id": execution_id,
                    "status": "error"
                }
            
            # Step 2: Execute multi-agent workflow
            results = await self._execute_multi_agent_workflow(
                execution_plan, session_id, execution_id
            )
            
            # Step 3: Aggregate and synthesize results
            final_response = await self._synthesize_results(
                user_query, execution_plan, results
            )
            
            # Update execution status
            self.active_executions[execution_id]["status"] = AgentStatus.COMPLETED
            self.active_executions[execution_id]["end_time"] = datetime.now()
            
            return {
                "response": final_response,
                "execution_id": execution_id,
                "execution_plan": execution_plan,
                "agent_results": results,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in orchestration: {str(e)}")
            if execution_id in self.active_executions:
                self.active_executions[execution_id]["status"] = AgentStatus.ERROR
                self.active_executions[execution_id]["error"] = str(e)
            
            return {
                "response": f"I encountered an error processing your request: {str(e)}",
                "execution_id": execution_id,
                "status": "error"
            }

    async def _create_execution_plan(self, user_query: str, context: Dict) -> Dict:
        """
        Use GPT-4.1 nano to create an intelligent execution plan
        """
        try:
            # Prepare context information
            context_info = ""
            if context:
                if context.get("has_statement"):
                    context_info += "- User has uploaded a bank statement for analysis\n"
                if context.get("available_factsheets"):
                    context_info += f"- {len(context['available_factsheets'])} factsheets available\n"
            
            planning_prompt = f"""
User Query: "{user_query}"

Available Context:
{context_info if context_info else "- No additional context available"}

Create an execution plan to answer this query effectively. Please provide your response as a JSON object following the specified format."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.orchestration_prompt},
                    {"role": "user", "content": planning_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            execution_plan = json.loads(response.choices[0].message.content)
            logger.info(f"Created execution plan: {execution_plan}")
            return execution_plan
            
        except Exception as e:
            logger.error(f"Error creating execution plan: {e}")
            return {}

    async def _execute_multi_agent_workflow(self, plan: Dict, session_id: str, execution_id: str) -> Dict:
        """
        Execute the planned workflow with multiple agents
        """
        results = {}
        agents_needed = plan.get("agents_needed", [])
        execution_sequence = plan.get("execution_sequence", [])
        
        # Get the original user query from execution tracking
        original_query = self.active_executions.get(execution_id, {}).get("query", "")
        
        # Update execution tracking
        self.active_executions[execution_id]["status"] = AgentStatus.EXECUTING
        self.active_executions[execution_id]["agents"] = [
            AgentExecution(agent, AgentStatus.IDLE, 0.0, "Waiting to start")
            for agent in agents_needed
        ]
        
        for step in execution_sequence:
            agent_name = step.get("agent")
            action = step.get("action")
            inputs = step.get("inputs", {})
            
            # Ensure the query is included in inputs if not present
            if "query" not in inputs and original_query:
                inputs["query"] = original_query
                logger.info(f"Added original query to inputs: {original_query}")
            
            if agent_name not in self.agents:
                logger.warning(f"Unknown agent: {agent_name}")
                continue
            
            # Update agent status
            self._update_agent_status(execution_id, agent_name, AgentStatus.EXECUTING, 0.0, f"Starting {action}")
            
            try:
                # Execute agent action
                if agent_name == "statement_analyst":
                    result = await self._execute_statement_analyst(action, inputs, session_id)
                elif agent_name == "factsheet_analyst":
                    result = await self._execute_factsheet_analyst(action, inputs, session_id)
                elif agent_name == "news_agent":
                    result = await self._execute_news_agent(action, inputs, session_id)
                else:
                    result = {"error": f"Unknown agent: {agent_name}"}
                
                results[agent_name] = result
                self._update_agent_status(execution_id, agent_name, AgentStatus.COMPLETED, 100.0, "Completed successfully")
                
            except Exception as e:
                error_msg = f"Error in {agent_name}: {str(e)}"
                logger.error(error_msg)
                results[agent_name] = {"error": error_msg}
                self._update_agent_status(execution_id, agent_name, AgentStatus.ERROR, 0.0, error_msg)
        
        return results

    async def _execute_statement_analyst(self, action: str, inputs: Dict, session_id: str) -> Dict:
        """Execute statement_analyst actions"""
        logger.info(f"Executing statement_analyst action: {action} with inputs: {inputs} for session: {session_id}")
        
        # Check if statement exists first
        if not self.agents["statement_analyst"].has_statement(session_id):
            logger.warning(f"No statement found for session: {session_id}")
            return {
                "error": "No statement found for this session. Please upload a statement first.",
                "status": "error"
            }
        
        if action == "query_statement":
            query = inputs.get("query", "")
            if not query:
                logger.error("No query provided in inputs")
                return {
                    "error": "No query provided",
                    "status": "error"
                }
            logger.info(f"Querying statement with: {query}")
            return await self.agents["statement_analyst"].query_statement(query, session_id)
        elif action == "get_positions":
            # Extract position data if available
            return await self.agents["statement_analyst"].query_statement("List all my investment positions with values", session_id)
        else:
            return {"error": f"Unknown statement_analyst action: {action}"}

    async def _execute_factsheet_analyst(self, action: str, inputs: Dict, session_id: str) -> Dict:
        """Execute factsheet_analyst actions"""
        logger.info(f"Executing factsheet_analyst action: {action} with inputs: {inputs}")
        
        if action == "search_factsheets":
            query = inputs.get("query", "")
            if not query:
                return {
                    "error": "No query provided for factsheet search",
                    "status": "error"
                }
            return await self.agents["factsheet_analyst"].search_factsheets(query, session_id)
        elif action == "analyze_fund_performance":
            fund_name = inputs.get("fund_name", "")
            return await self.agents["factsheet_analyst"].analyze_fund_performance(fund_name, session_id)
        elif action == "compare_investments":
            fund_names = inputs.get("fund_names", [])
            return await self.agents["factsheet_analyst"].compare_investments(fund_names, session_id)
        else:
            return {"error": f"Unknown factsheet analyst action: {action}"}

    async def _execute_news_agent(self, action: str, inputs: Dict, session_id: str) -> Dict:
        """
        Execute news agent actions
        
        Why this structure:
        1. Supports different types of news queries
        2. Flexible input handling for various scenarios
        3. Error handling for API failures
        """
        try:
            if action == "get_asset_news":
                assets = inputs.get("assets", [])
                days_back = inputs.get("days_back", 7)
                
                if not assets:
                    return {"error": "No assets provided for news search"}
                
                return await self.agents["news_agent"].get_asset_news(assets, session_id, days_back)
                
            elif action == "get_market_news":
                # General market news query
                market_terms = inputs.get("terms", ["market", "economy", "finance"])
                return await self.agents["news_agent"].get_asset_news(market_terms, session_id)
                
            else:
                return {"error": f"Unknown news agent action: {action}"}
                
        except Exception as e:
            logger.error(f"Error executing news agent: {e}")
            return {"error": str(e)}

    async def _synthesize_results(self, user_query: str, plan: Dict, results: Dict) -> str:
        """
        Synthesize results from multiple agents into a cohesive response
        """
        try:
            # Prepare results summary for GPT-4
            results_summary = []
            for agent_name, result in results.items():
                if "error" not in result:
                    results_summary.append(f"**{agent_name.replace('_', ' ').title()}:**\n{json.dumps(result, indent=2)}")
                else:
                    results_summary.append(f"**{agent_name.replace('_', ' ').title()}:** Error - {result['error']}")
            
            synthesis_prompt = f"""
User asked: "{user_query}"

Results from specialized agents:
{chr(10).join(results_summary)}

Synthesize these results into a comprehensive, helpful response for the user. 
- Focus on answering their specific question
- Highlight key insights and actionable information
- Use a conversational, professional tone
- Include specific numbers, dates, and details when available
- If any agent encountered errors, mention limitations gracefully
"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial AI assistant that synthesizes information from multiple sources to provide comprehensive answers."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error synthesizing results: {e}")
            return "I gathered information from multiple sources but encountered an issue creating the final response. Please try asking a more specific question."

    def _update_agent_status(self, execution_id: str, agent_name: str, status: AgentStatus, progress: float, message: str):
        """Update the status of a specific agent in the execution"""
        if execution_id in self.active_executions:
            agents = self.active_executions[execution_id]["agents"]
            for agent in agents:
                if agent.agent_name == agent_name:
                    agent.status = status
                    agent.progress = progress
                    agent.message = message
                    break

    def get_execution_status(self, execution_id: str) -> Optional[Dict]:
        """Get current execution status for frontend updates"""
        if execution_id not in self.active_executions:
            return None
        
        execution = self.active_executions[execution_id]
        return {
            "execution_id": execution_id,
            "status": execution["status"].value if isinstance(execution["status"], AgentStatus) else str(execution["status"]),
            "query": execution["query"],
            "agents": [
                {
                    "name": agent.agent_name,
                    "status": agent.status.value,
                    "progress": agent.progress,
                    "message": agent.message
                } for agent in execution.get("agents", [])
            ],
            "start_time": execution["start_time"].isoformat() if "start_time" in execution else None,
            "end_time": execution.get("end_time", {}).isoformat() if "end_time" in execution else None
        }

# Test the orchestrator
async def test_orchestrator():
    """Test the orchestrator with a sample query"""
    orchestrator = FinancialOrchestrator()
    
    # Test query
    result = await orchestrator.process_query(
        "What are my top 3 investment positions and can you find recent news about them?",
        "test_session",
        {"has_statement": True}
    )
    
    print("Orchestrator Test Result:")
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_orchestrator())
