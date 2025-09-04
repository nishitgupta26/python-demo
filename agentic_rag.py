"""
Agentic RAG System
=======================

A proper agentic RAG system with LLM orchestrator that can:
- Plan and reason about queries
- Decide when to use RAG vs MCP tools
- Iterate and refine based on results
- Execute multi-step workflows

Architecture:
[User Query] → [LLM Orchestrator] → [Plan] → [RAG/MCP] → [Reason] → [Iterate] → [Final Answer]

Usage:
    from agentic_rag import AgenticRAG
    
    rag = AgenticRAG()
    rag.process_pdfs()
    
    # True agentic query with planning and iteration
    result = rag.query("What is JavaScript and find recent updates about it?")
    print(result['response'])
"""

import os
import json
import logging
import wikipedia
from typing import Dict, Any, List, Optional
from datetime import datetime
from simple_rag_system import SimpleRAGSystem

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WikipediaTool:
    """MCP-style Wikipedia tool for external knowledge."""
    
    def __init__(self):
        try:
            wikipedia.set_lang("en")
            self.available = True
            logger.info("🌐 Wikipedia MCP tool initialized")
        except:
            self.available = False
            logger.warning("⚠️ Wikipedia MCP tool unavailable")
    
    def search_wikipedia(self, query: str, max_results: int = 2) -> Dict[str, Any]:
        """MCP tool function: Search Wikipedia."""
        if not self.available:
            return {"success": False, "error": "Wikipedia not available", "results": []}
        0
        try:
            search_results = wikipedia.search(query, results=max_results)
            results = []
            
            for title in search_results:
                try:
                    summary = wikipedia.summary(title, sentences=3)
                    page = wikipedia.page(title)
                    results.append({
                        "title": title,
                        "content": summary,
                        "url": page.url
                    })
                    logger.info(f"🔧 MCP Wikipedia: Found {search_results}")
                except:
                    continue
            
            logger.info(f"🔧 MCP Wikipedia: Found {len(results)} articles")
            return {"success": True, "results": results, "tool": "wikipedia_search"}
        except Exception as e:
            return {"success": False, "error": str(e), "results": []}


class AgentOrchestrator:
    """LLM-based orchestrator that plans, reasons, and iterates."""
    
    def __init__(self, base_rag: SimpleRAGSystem, mcp_tools: Dict):
        self.base_rag = base_rag
        self.mcp_tools = mcp_tools
        self.max_iterations = 3
        self.execution_log = []
    
    def _call_llm_for_planning(self, query: str, context: str = "") -> Dict[str, Any]:
        """Use LLM to create execution plan."""
        planning_prompt = f"""You are an intelligent agent with access to local documents and external knowledge.

USER QUERY: {query}

Planning Strategy:
Analyze the user's query intelligently and decide what sources will provide the best answer.

Core Principles:
1. Local documents (RAG) contain your primary knowledge - always consider them first
2. External knowledge (Wikipedia) provides broader context and recent information
3. Use your intelligence to determine what combination of sources will best serve the user

Decision Framework:
- Analyze what the user is asking for
- Consider if local documents likely contain relevant information
- Determine if external sources would add value or fill knowledge gaps
- Plan the most efficient path to a comprehensive answer

Be intelligent: There are no rigid rules - use your reasoning to determine the best approach for each unique query.

Respond ONLY in this format:
NEEDS_RAG: yes/no
NEEDS_WIKIPEDIA: yes/no  
REASONING: brief explanation of strategy
RAG_QUERY: specific query for local search
WIKIPEDIA_QUERY: specific query for Wikipedia (if needed)"""
        
        try:
            response = self.base_rag.llm.invoke(planning_prompt)
            plan_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse the structured response
            needs_rag = "needs_rag: yes" in plan_text.lower()
            needs_wikipedia = "needs_wikipedia: yes" in plan_text.lower()
            
            # Extract reasoning
            reasoning = "Local search planned"
            if "reasoning:" in plan_text.lower():
                reasoning_part = plan_text.lower().split("reasoning:")[1].split("\n")[0].strip()
                if reasoning_part:
                    reasoning = reasoning_part
            
            # Extract queries
            rag_query = query  # default
            if "rag_query:" in plan_text.lower():
                rag_part = plan_text.lower().split("rag_query:")[1].split("\n")[0].strip()
                if rag_part:
                    rag_query = rag_part
            
            wikipedia_query = query  # default
            if "wikipedia_query:" in plan_text.lower():
                wiki_part = plan_text.lower().split("wikipedia_query:")[1].split("\n")[0].strip()
                if wiki_part:
                    wikipedia_query = wiki_part
            
            # Build steps based on LLM decisions
            steps = []
            
            # Always prioritize RAG for any query that could benefit from local documents
            if needs_rag:
                steps.append({"action": "RAG_SEARCH", "query": rag_query})
            
            if needs_wikipedia:
                steps.append({"action": "WIKIPEDIA_SEARCH", "query": wikipedia_query})
            
            # CRITICAL: Ensure at least RAG search - local knowledge is always valuable
            if not steps:
                steps.append({"action": "RAG_SEARCH", "query": query})
                needs_rag = True
            
            # Trust the LLM's decision - no hardcoded keyword matching
            
            return {
                "needs_rag": needs_rag,
                "needs_mcp": needs_wikipedia,
                "reasoning": reasoning,
                "steps": steps
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Planning failed: {e}, using fallback")
            return {
                "needs_rag": True,
                "needs_mcp": False,
                "reasoning": "Fallback plan due to planning error",
                "steps": [{"action": "RAG_SEARCH", "query": query}]
            }
    
    def _execute_rag_search(self, query: str) -> Dict[str, Any]:
        """Execute RAG search step."""
        logger.info(f"🔍 Executing RAG search: '{query}'")
        
        try:
            docs = self.base_rag.retrieve_data(query, num_results=4)
            
            if docs:
                # Extract content for reasoning
                content = []
                sources = []
                for doc in docs:
                    content.append(doc.page_content)
                    sources.append(doc.metadata.get('filename', 'Unknown'))
                
                result = {
                    "success": True,
                    "content": content,
                    "sources": sources,
                    "count": len(docs),
                    "tool": "rag_search"
                }
                logger.info(f"✅ RAG search: Found {len(docs)} documents")
                return result
            else:
                return {
                    "success": False,
                    "content": [],
                    "sources": [],
                    "count": 0,
                    "error": "No relevant documents found",
                    "tool": "rag_search"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content": [],
                "sources": [],
                "count": 0,
                "tool": "rag_search"
            }
    
    def _execute_mcp_action(self, action: str, query: str) -> Dict[str, Any]:
        """Execute MCP tool action."""
        logger.info(f"🔧 Executing MCP action: {action} with query: '{query}'")
        
        if action == "WIKIPEDIA_SEARCH" and "wikipedia" in self.mcp_tools:
            return self.mcp_tools["wikipedia"].search_wikipedia(query)
        else:
            return {
                "success": False,
                "error": f"MCP tool {action} not available",
                "tool": action.lower()
            }
    
    def _reason_and_decide(self, query: str, results: List[Dict], iteration: int) -> Dict[str, Any]:
        """Use LLM to reason about results and decide next steps."""
        
        # Prepare context from results
        context_parts = []
        for result in results:
            if result.get("success"):
                if result.get("tool") == "rag_search":
                    context_parts.append(f"RAG Results: Found {result['count']} documents")
                    context_parts.extend([f"- {content[:200]}..." for content in result.get("content", [])[:2]])
                elif result.get("tool") == "wikipedia_search":
                    context_parts.append(f"Wikipedia Results: Found {len(result.get('results', []))} articles")
                    for article in result.get("results", [])[:2]:
                        context_parts.append(f"- {article['title']}: {article['content'][:200]}...")
        
        context = "\n".join(context_parts)
        
        reasoning_prompt = f"""You are an intelligent agent. Analyze the results and decide if more information is needed.

ORIGINAL QUERY: {query}
ITERATION: {iteration}/{self.max_iterations}

CURRENT INFORMATION GATHERED:
{context}

Think intelligently:
- Can you provide a comprehensive answer with the current information?
- Are there important aspects of the query that remain unanswered?
- Would additional sources significantly improve the response quality?
- What would be the most valuable next step if more information is needed?

Respond ONLY in this format:
SUFFICIENT: yes/no
REASONING: your intelligent analysis
NEXT_ACTION: COMPLETE/RAG_SEARCH/WIKIPEDIA_SEARCH
CONFIDENCE: 0.1-1.0

Analysis:"""
        
        try:
            response = self.base_rag.llm.invoke(reasoning_prompt)
            analysis_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse the structured response
            sufficient = "sufficient: yes" in analysis_text.lower()
            
            # Extract reasoning
            reasoning = "The current results are sufficient to answer the query."
            if "reasoning:" in analysis_text.lower():
                reasoning_part = analysis_text.lower().split("reasoning:")[1].split("\n")[0].strip()
                if reasoning_part:
                    reasoning = reasoning_part
            
            # Extract next action
            next_action = "COMPLETE"
            if "next_action:" in analysis_text.lower():
                action_part = analysis_text.lower().split("next_action:")[1].split("\n")[0].strip()
                if "rag_search" in action_part:
                    next_action = "RAG_SEARCH"
                elif "wikipedia_search" in action_part:
                    next_action = "WIKIPEDIA_SEARCH"
            
            # Extract confidence
            confidence = 0.7
            if "confidence:" in analysis_text.lower():
                conf_part = analysis_text.lower().split("confidence:")[1].split("\n")[0].strip()
                try:
                    confidence = float(conf_part)
                except:
                    confidence = 0.7
            
            logger.info(f"🧠 Reasoning: {reasoning}")
            return {
                "sufficient": sufficient or iteration >= self.max_iterations,
                "reasoning": reasoning,
                "next_action": next_action if not sufficient else "COMPLETE",
                "refined_query": query,
                "confidence": confidence
            }
        except Exception as e:
            logger.warning(f"⚠️ Reasoning failed: {e}")
            return {
                "sufficient": True,
                "reasoning": "Fallback: proceeding with available results",
                "next_action": "COMPLETE",
                "confidence": 0.5
            }
    
    def _verify_content_consistency(self, rag_content: List[str], mcp_content: List[str], query: str) -> Dict[str, Any]:
        """Simple verification to check for contradictions between sources."""
        if not rag_content or not mcp_content:
            return {"has_contradictions": False, "confidence_adjustment": 0.0, "notes": "Single source - no cross-verification needed"}
        
        # Create verification prompt with simpler format
        rag_summary = " ".join(rag_content[:2])[:400]  # Limit length
        mcp_summary = " ".join(mcp_content[:2])[:400]
        
        verification_prompt = f"""Compare these sources about "{query}" for contradictions.

LOCAL: {rag_summary}

EXTERNAL: {mcp_summary}

Are they consistent? Answer ONLY with this format:
CONSISTENT: yes/no
ISSUES: brief description or "none"
CONFIDENCE: high/medium/low"""
        
        try:
            response = self.base_rag.llm.invoke(verification_prompt)
            verification_text = response.content if hasattr(response, 'content') else str(response)
            
            # Simple text parsing instead of JSON
            consistent = "yes" in verification_text.lower() and "consistent:" in verification_text.lower()
            has_contradictions = not consistent
            
            # Extract confidence level
            confidence_level = "medium"  # default
            if "confidence: high" in verification_text.lower():
                confidence_level = "high"
            elif "confidence: low" in verification_text.lower():
                confidence_level = "low"
            
            # Extract issues
            issues = "No issues found"
            if "issues:" in verification_text.lower():
                issues_part = verification_text.lower().split("issues:")[1].split("\n")[0].strip()
                if issues_part and issues_part != "none":
                    issues = issues_part
            
            # Calculate confidence adjustment
            adjustment = 0.0
            if has_contradictions:
                adjustment = -0.3  # Reduce confidence if contradictions found
            elif confidence_level == "high":
                adjustment = 0.1   # Boost if sources agree strongly
            elif confidence_level == "low":
                adjustment = -0.1  # Slight reduction for low confidence
            
            logger.info(f"🔍 Verification: {issues}")
            return {
                "has_contradictions": has_contradictions,
                "confidence_adjustment": adjustment,
                "notes": issues,
                "recommendation": "use both" if consistent else "use with caution"
            }
                
        except Exception as e:
            logger.warning(f"⚠️ Verification failed: {e}")
            return {"has_contradictions": False, "confidence_adjustment": -0.1, "notes": "Verification unavailable"}
    
    def _synthesize_final_answer(self, query: str, all_results: List[Dict]) -> Dict[str, Any]:
        """Generate final answer by synthesizing all results with verification."""
        
        # Collect all information
        rag_content = []
        rag_sources = []
        mcp_content = []
        mcp_sources = []
        
        for result in all_results:
            if result.get("success"):
                if result.get("tool") == "rag_search":
                    rag_content.extend(result.get("content", []))
                    rag_sources.extend(result.get("sources", []))
                elif result.get("tool") == "wikipedia_search":
                    for article in result.get("results", []):
                        mcp_content.append(f"{article['title']}: {article['content']}")
                        mcp_sources.append(article['title'])
        
        # Verify content consistency
        verification = self._verify_content_consistency(rag_content, mcp_content, query)
        
        # Create synthesis prompt with verification awareness
        context_parts = []
        if rag_content:
            context_parts.append("LOCAL DOCUMENTS:")
            context_parts.extend([f"- {content}" for content in rag_content[:3]])
        
        if mcp_content:
            context_parts.append("\nEXTERNAL SOURCES:")
            context_parts.extend([f"- {content}" for content in mcp_content[:3]])
        
        # Add verification notes if there are issues
        if verification.get("notes"):
            context_parts.append(f"\nVERIFICATION NOTES: {verification['notes']}")
        
        context = "\n".join(context_parts) if context_parts else "No relevant information found."
        
        # User-friendly synthesis prompt
        synthesis_prompt = f"""You are a knowledgeable assistant. Provide a comprehensive, user-friendly answer.

USER QUERY: {query}

AVAILABLE INFORMATION:
{context}

Guidelines:
- Give a complete, natural answer that directly addresses the user's question
- Combine information from all sources seamlessly 
- Don't mention "local documents" or "external sources" - just provide the answer
- If there are contradictions, present the most accurate information
- Be conversational and helpful
- Focus on what the user wants to know, not where the information came from
- Make the response flow naturally as if from a knowledgeable expert

Answer:"""
        
        try:
            response = self.base_rag.llm.invoke(synthesis_prompt)
            final_answer = response.content if hasattr(response, 'content') else str(response)
            
            # Calculate confidence based on available information and verification
            confidence = 0.3  # Base confidence
            if rag_content:
                confidence += 0.4
            if mcp_content:
                confidence += 0.3
            
            # Apply verification adjustment
            confidence += verification.get("confidence_adjustment", 0.0)
            confidence = max(0.0, min(confidence, 1.0))  # Clamp between 0 and 1
            
            return {
                "response": final_answer,
                "confidence": confidence,
                "sources": list(set(rag_sources + mcp_sources)),
                "rag_sources": len(rag_sources),
                "mcp_sources": len(mcp_sources),
                "verification": verification,
                "execution_log": self.execution_log.copy()
            }
            
        except Exception as e:
            return {
                "response": f"I encountered an error while generating the response: {e}",
                "confidence": 0.0,
                "sources": [],
                "rag_sources": 0,
                "mcp_sources": 0,
                "verification": {"has_contradictions": False, "confidence_adjustment": 0.0, "notes": "Error during synthesis"},
                "execution_log": self.execution_log.copy()
            }
    
    def execute_agentic_query(self, query: str) -> Dict[str, Any]:
        """Main agentic execution loop with planning, reasoning, and iteration."""
        logger.info(f"🤖 Starting agentic execution for: '{query}'")
        
        self.execution_log = []
        all_results = []
        
        # Step 1: Initial Planning
        plan = self._call_llm_for_planning(query)
        self.execution_log.append({"step": "planning", "plan": plan})
        logger.info(f"📋 Plan: {plan.get('reasoning', 'No reasoning provided')}")
        
        # Step 2-5: Iterative Execution
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"🔄 Iteration {iteration}/{self.max_iterations}")
            
            iteration_results = []
            
            # Execute planned steps
            for step in plan.get("steps", []):
                if not step:  # Skip None steps
                    continue
                    
                action = step.get("action")
                step_query = step.get("query", query)
                
                if action == "RAG_SEARCH":
                    result = self._execute_rag_search(step_query)
                    iteration_results.append(result)
                    all_results.append(result)
                    
                elif action in ["WIKIPEDIA_SEARCH"]:
                    result = self._execute_mcp_action(action, step_query)
                    iteration_results.append(result)
                    all_results.append(result)
            
            self.execution_log.append({
                "step": f"iteration_{iteration}",
                "results": [{"tool": r.get("tool"), "success": r.get("success")} for r in iteration_results]
            })
            
            # Step 3: Reason and decide if we need more iterations
            if iteration < self.max_iterations:
                decision = self._reason_and_decide(query, all_results, iteration)
                self.execution_log.append({"step": "reasoning", "decision": decision})
                
                # Let the LLM's reasoning decide if more information is needed
                # No hardcoded keyword detection - trust the agent's intelligence
                
                if decision.get("sufficient", True):
                    logger.info("✅ Agent determined sufficient information gathered")
                    break
                else:
                    # Plan next iteration based on reasoning
                    next_action = decision.get("next_action")
                    refined_query = decision.get("refined_query", query)
                    
                    if next_action and next_action != "COMPLETE":
                        plan = {
                            "steps": [{"action": next_action, "query": refined_query}],
                            "reasoning": f"Iteration {iteration + 1}: {decision.get('reasoning')}"
                        }
                        logger.info(f"🔄 Continuing with: {next_action}")
                    else:
                        break
        
        # Step 6: Final Synthesis
        logger.info("🎯 Synthesizing final answer...")
        final_result = self._synthesize_final_answer(query, all_results)
        
        self.execution_log.append({"step": "synthesis", "completed": True})
        final_result["mode"] = "agentic"
        final_result["iterations"] = len([log for log in self.execution_log if log["step"].startswith("iteration")])
        
        logger.info(f"✅ Agentic execution completed: {final_result['confidence']:.2f} confidence")
        return final_result


class AgenticRAG:
    """
    Agentic RAG System with LLM Orchestrator.
    
    Features an intelligent agent that can plan, reason, iterate, and use tools.
    """
    
    def __init__(self, use_agents=True):
        """Initialize the Agentic RAG system."""
        logger.info("🚀 Initializing Agentic RAG System...")
        
        # Initialize base RAG system
        self.base_rag = SimpleRAGSystem()
        
        # Initialize MCP tools
        self.mcp_tools = {
            "wikipedia": WikipediaTool()
        }
        
        # Initialize agent orchestrator
        self.use_agents = use_agents
        if use_agents:
            self.orchestrator = AgentOrchestrator(self.base_rag, self.mcp_tools)
            logger.info("🤖 Agentic orchestrator initialized with planning and reasoning")
        else:
            self.orchestrator = None
            logger.info("📖 Simple RAG mode (no agents)")
    
    def process_pdfs(self) -> bool:
        """Process PDFs using the base RAG system."""
        return self.base_rag.process_all_pdfs()
    
    def query(self, query: str) -> Dict[str, Any]:
        """
        Process query with true agentic capabilities.
        
        The agent will:
        1. Plan the approach
        2. Execute RAG/MCP tools as needed  
        3. Reason about results
        4. Iterate if necessary
        5. Synthesize final answer
        """
        if not self.use_agents or not self.orchestrator:
            return self._simple_query(query)
        
        return self.orchestrator.execute_agentic_query(query)
    
    def _simple_query(self, query: str) -> Dict[str, Any]:
        """Fallback to simple RAG."""
        try:
            docs = self.base_rag.retrieve_data(query)
            response = self.base_rag.generate_response(query, docs)
            sources = [doc.metadata.get('filename', 'Unknown') for doc in docs] if docs else []
            
            return {
                "response": response,
                "confidence": 0.8 if docs else 0.3,
                "sources": sources,
                "mode": "simple_rag",
                "rag_sources": len(sources),
                "mcp_sources": 0,
                "verification": {"has_contradictions": False, "confidence_adjustment": 0.0, "notes": "Simple RAG mode - no verification"},
                "iterations": 1,
                "execution_log": [{"step": "simple_rag", "completed": True}]
            }
        except Exception as e:
            return {
                "response": f"Error: {e}",
                "confidence": 0.0,
                "sources": [],
                "mode": "error",
                "rag_sources": 0,
                "mcp_sources": 0,
                "verification": {"has_contradictions": False, "confidence_adjustment": 0.0, "notes": "Error occurred"},
                "iterations": 0,
                "execution_log": [{"step": "error", "error": str(e)}]
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        base_info = self.base_rag.get_system_info()
        
        agentic_info = {
            "system_type": "true_agentic_rag" if self.use_agents else "simple_rag",
            "agentic_mode": self.use_agents,
            "mcp_tools": list(self.mcp_tools.keys()),
            "wikipedia_available": self.mcp_tools["wikipedia"].available,
            "max_iterations": self.orchestrator.max_iterations if self.orchestrator else 1
        }
        
        return {**base_info, **agentic_info}


def main():
    """Demo of the agentic RAG system."""
    print("🤖 Agentic RAG System")
    print("=" * 50)
    
    try:
        # Initialize system
        rag = AgenticRAG(use_agents=True)
        
        # Show system info
        info = rag.get_system_info()
        print(f"\n📊 System Status:")
        print(f"   Type: {info['system_type']}")
        print(f"   Documents: {info['document_count']}")
        print(f"   MCP Tools: {info['mcp_tools']}")
        print(f"   Max Iterations: {info['max_iterations']}")
        
        # Process PDFs
        print(f"\n📄 Processing PDFs...")
        success = rag.process_pdfs()
        
        if success:
            info = rag.get_system_info()
            print(f"✅ Total documents: {info['document_count']}")
            
            # Example agentic queries for different scenarios and domains
            queries = [
                "What is Javascript?",
                "Explain about accelerators and GPUs",
                "What is the difference between JavaScript and C++?"
            ]
            
            for query in queries:
                print(f"\n🔍 Agentic Query: '{query}'")
                print("-" * 60)
                
                result = rag.query(query)
                
                # User-friendly output
                print(f"🤖 Agent Response:")
                print(f"Confidence: {result['confidence']:.1f}/1.0")
                print(f"Sources Used: {result['rag_sources'] + result['mcp_sources']} total")
                
                print(f"\n💬 Answer:")
                print(result['response'])
                
                # Technical details (for development)
                print(f"\n🔧 Technical Details:")
                print(f"   Mode: {result['mode']}")
                print(f"   RAG Sources: {result['rag_sources']}, External Sources: {result['mcp_sources']}")
                print(f"   Iterations: {result['iterations']}")
                
                # Show verification if there are issues
                verification = result.get('verification', {})
                if verification.get('has_contradictions'):
                    print(f"   ⚠️ Note: Some contradictions detected between sources")
                elif verification.get('notes') and 'unavailable' not in verification.get('notes', ''):
                    print(f"   ✅ Sources verified as consistent")
                
                print("-" * 60)
        
        else:
            print("⚠️ No PDFs found. Add PDF files to the 'pdfs' folder.")
    
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()