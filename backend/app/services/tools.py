"""Agentic tools — web search, calculator, document summarizer, query planning (Phase 5)."""

import logging
import re
import ast
import operator
from typing import Optional

import httpx

from app.config import settings
from app.services.vectorstore import hybrid_search, get_db

logger = logging.getLogger(__name__)

# Safe operators for calculator
SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


# ── Web Search Tool ───────────────────────────────────────────────

async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Web search tool for when internal documents lack the answer.
    Uses DuckDuckGo Instant Answer API (no API key required).
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1},
            )
            data = response.json()

        results = []

        # Abstract (top answer)
        if data.get("Abstract"):
            results.append({
                "title": data.get("Heading", ""),
                "snippet": data["Abstract"],
                "url": data.get("AbstractURL", ""),
                "source": data.get("AbstractSource", ""),
            })

        # Related topics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": topic.get("Text", "")[:80],
                    "snippet": topic.get("Text", ""),
                    "url": topic.get("FirstURL", ""),
                    "source": "DuckDuckGo",
                })

        logger.info(f"Web search for '{query}': {len(results)} results")
        return results[:max_results]

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return [{"title": "Error", "snippet": f"Web search failed: {str(e)}", "url": "", "source": ""}]


# ── Calculator / Code Interpreter Tool ────────────────────────────

def _safe_eval_node(node):
    """Safely evaluate an AST math expression node."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value}")
    elif isinstance(node, ast.UnaryOp):
        op_func = SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary op: {type(node.op)}")
        return op_func(_safe_eval_node(node.operand))
    elif isinstance(node, ast.BinOp):
        op_func = SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported binary op: {type(node.op)}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return op_func(left, right)
    else:
        raise ValueError(f"Unsupported expression type: {type(node)}")


async def calculate(expression: str) -> dict:
    """
    Safe calculator tool. Evaluates mathematical expressions without exec/eval.
    Supports: +, -, *, /, **, %, //
    """
    try:
        # Clean the expression
        cleaned = expression.strip()
        cleaned = re.sub(r'[^\d+\-*/%.() ]', '', cleaned)

        if not cleaned:
            return {"expression": expression, "result": None, "error": "Empty expression"}

        tree = ast.parse(cleaned, mode="eval")
        result = _safe_eval_node(tree.body)

        return {
            "expression": expression,
            "result": result,
            "error": None,
        }
    except ZeroDivisionError:
        return {"expression": expression, "result": None, "error": "Division by zero"}
    except Exception as e:
        return {"expression": expression, "result": None, "error": str(e)}


# ── Document Summarizer Tool ─────────────────────────────────────

async def summarize_document(doc_id: str) -> dict:
    """
    Generate a full document summary, separate from Q&A chat.
    Fetches all chunks for the document and asks the LLM to synthesize.
    """
    from app.services.llm import generate_text

    conn = get_db()

    # Get document metadata
    doc_row = conn.execute(
        "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    if doc_row is None:
        return {"doc_id": doc_id, "error": "Document not found", "summary": ""}

    doc = dict(doc_row)

    # Get all chunks for this document
    chunk_rows = conn.execute(
        "SELECT content, page_number FROM chunks WHERE doc_id = ? ORDER BY page_number, chunk_id",
        (doc_id,),
    ).fetchall()

    if not chunk_rows:
        return {"doc_id": doc_id, "error": "No chunks found", "summary": ""}

    # Build document text (limit to avoid token overflow)
    doc_text = ""
    for row in chunk_rows:
        doc_text += f"\n[Page {row['page_number']}]\n{row['content']}\n"
        if len(doc_text) > 8000:  # Keep under typical context limit
            doc_text += "\n... (truncated)"
            break

    prompt = f"""Summarize the following document comprehensively. Include:
1. Main topic and purpose
2. Key points and findings
3. Important details, data, or conclusions
4. Document structure overview

Document: {doc['filename']}

Content:
{doc_text}

Provide a well-structured summary in markdown format."""

    try:
        summary = await generate_text(prompt)
        return {
            "doc_id": doc_id,
            "filename": doc["filename"],
            "total_chunks": len(chunk_rows),
            "summary": summary,
            "error": None,
        }
    except Exception as e:
        logger.error(f"Summarization failed for {doc_id}: {e}")
        return {
            "doc_id": doc_id,
            "filename": doc.get("filename", ""),
            "summary": "",
            "error": str(e),
        }


# ── Query Planning Tool ──────────────────────────────────────────

async def plan_and_execute_query(query: str, filters: dict = None) -> dict:
    """
    For multi-step/comparative questions: break the question into sub-steps,
    retrieve for each, and synthesize a final answer.
    """
    from app.services.llm import generate_text

    # Step 1: Ask LLM to decompose the query
    planning_prompt = f"""Break the following complex question into 2-4 simple sub-questions that can be answered independently from a document database.
Return ONLY the sub-questions, one per line, numbered. No explanations.

Question: {query}"""

    try:
        text = await generate_text(planning_prompt)

        sub_questions = []
        for line in text.strip().split("\n"):
            cleaned = line.strip()
            if cleaned and cleaned[0].isdigit():
                cleaned = cleaned.lstrip("0123456789.)- ").strip()
            if cleaned and len(cleaned) > 5:
                sub_questions.append(cleaned)

        if not sub_questions:
            sub_questions = [query]

    except Exception as e:
        logger.error(f"Query planning failed: {e}")
        sub_questions = [query]

    # Step 2: Retrieve for each sub-question
    all_chunks = {}
    sub_results = []

    for sq in sub_questions[:4]:  # Max 4 sub-questions
        results = await hybrid_search(sq, top_k=5, filters=filters)
        for chunk in results:
            cid = chunk.get("chunk_id", "")
            if cid not in all_chunks or chunk.get("score", 0) > all_chunks[cid].get("score", 0):
                all_chunks[cid] = chunk
        sub_results.append({"question": sq, "num_results": len(results)})

    # Step 3: Synthesize with all gathered context
    context_chunks = sorted(all_chunks.values(), key=lambda x: x.get("score", 0), reverse=True)[:15]

    context_parts = []
    for i, chunk in enumerate(context_chunks):
        source = f"[Source {i+1}: {chunk.get('filename', '?')}, Page {chunk.get('page_number', '?')}]"
        context_parts.append(f"{source}\n{chunk.get('content', '')}")

    context_str = "\n\n---\n\n".join(context_parts)

    synthesis_prompt = f"""You were asked a complex question that was broken into sub-questions.

Original question: {query}

Sub-questions explored:
{chr(10).join(f'- {sq}' for sq in sub_questions)}

Context gathered from documents:
{context_str}

Synthesize a comprehensive answer to the original question using all the context above.
Cite sources using [Source: filename, Page X] format."""

    try:
        answer = await generate_text(
            synthesis_prompt,
            system="You are a helpful document analysis assistant. Answer based only on provided context.",
        )

        return {
            "query": query,
            "sub_questions": sub_results,
            "answer": answer,
            "sources": [
                {
                    "filename": c.get("filename", ""),
                    "page_number": c.get("page_number", 1),
                    "chunk_id": c.get("chunk_id", ""),
                    "score": c.get("score", 0),
                }
                for c in context_chunks[:5]
            ],
            "error": None,
        }
    except Exception as e:
        logger.error(f"Query synthesis failed: {e}")
        return {
            "query": query,
            "sub_questions": sub_results,
            "answer": "",
            "sources": [],
            "error": str(e),
        }
