from flask import Flask, render_template, request, jsonify
import requests
import subprocess
import psycopg2
import json
import re
from rag_utils import get_context, get_embedding, save_message, get_connection, chunk_text
from knowledge_extractor import extract_structured_knowledge
import os
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import docx
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
from knowledge_graph import KnowledgeGraph

model = SentenceTransformer("all-mpnet-base-v2")

load_dotenv()
app = Flask(__name__)

# --- CONFIG ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = os.getenv("GEMINI_URL")

# --- ROUTES ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message")
    model = data.get("model")  # "gemini" or "ollama"
    use_rag = data.get("use_rag")
    use_structure = data.get("use_structure", False)  # New parameter
    
    save_message("user", user_input)

    prompt_lower = user_input.lower()
    if any(word in prompt_lower for word in ["quiz", "question", "test", "mcq"]):
        task = "quiz"
        # use_rag = True
    elif any(word in prompt_lower for word in ["implement", "practice", "code", "program", "exercise"]):
        task = "practice"
        # use_rag = True
    elif any(word in prompt_lower for word in ["review", "explain", "summarize", "summary"]):
        task = "review"
        # use_rag = True
    else:
        task = "general"
        # use_rag = False

    rag_context = get_context(user_input) if use_rag else ""

    # Get Knowledge Graph context
    structured_context = ""
    if use_structure:
        try:
            kg = KnowledgeGraph()
            search_terms = user_input.lower().split()
            structured_context = kg.get_concept_context(search_terms)
            kg.close()
            print(f"Graph context found: {len(structured_context)} characters")
        except Exception as e:
            print(f"Knowledge graph query failed: {e}")

    # --- Prompt Engineering ---
    if task == "review":
        task_prompt = f"Review the following material and explain it simply:\n{user_input}"
    elif task == "quiz":
        task_prompt = (
            "Generate 5 multiple-choice quiz questions in the following JSON format:\n"
            '[{"question": "...", "options": ["A) Full option text", "B) Full option text", "C) Full option text", "D) Full option text"], "answer": "A) Full option text"}, ...]\n'
            f"Material:\n{user_input}\n"
            "Make sure the answer field contains the full option text including the letter prefix."
        )
    elif task == "practice":
        task_prompt = (
            "Give me a practical implementation exercise (with a brief solution):\n"
            f"{user_input}"
        )
    else:
        task_prompt = user_input
        
    
    # Enhanced system instruction for graph-aware responses
    system_instruction = ""
    if use_structure and structured_context:
        system_instruction = """You are an AI Teaching Assistant with access to a knowledge graph. 
        Use the graph relationships to provide comprehensive explanations that show how concepts connect. 
        Structure your response with clear headings and explain concept relationships."""

    # Combine contexts
    final_prompt = f"""
    {system_instruction}
    
    Knowledge Graph Context:
    {structured_context}
    
    RAG Context:
    {rag_context}
    
    Task: {task_prompt}
    """
    
    # Limit prompt size
    MAX_PROMPT_CHARS = 4000
    if len(final_prompt) > MAX_PROMPT_CHARS:
        final_prompt = final_prompt[:MAX_PROMPT_CHARS] + "\n...[truncated]..."


    if model == "gemini":
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": GEMINI_API_KEY
        }
        payload = {
            "contents": [{"parts": [{"text": final_prompt}]}]
        }
        response = requests.post(GEMINI_URL, headers=headers, data=json.dumps(payload))
        try:
            data = response.json()
            if "candidates" in data and data["candidates"]:
                reply = data["candidates"][0]["content"]["parts"][0]["text"]
            elif "error" in data:
                reply = f"Gemini API error: {data['error'].get('message', 'Unknown error')}"
            else:
                reply = "Gemini API returned an unexpected response."
        except Exception as e:
            reply = f"Failed to parse Gemini API response: {str(e)}"

    elif model == "ollama":
        result = subprocess.run(
            ["ollama", "run", "llama3:8b"],
            input=final_prompt,
            text=True,
            encoding="utf-8",
            capture_output=True
        )
        reply = result.stdout.strip()

    else:
        reply = "Invalid model selection."
    
    if task == "quiz":
        # Try to extract JSON from the reply
        match = re.search(r"\[.*\]", reply, re.DOTALL)
        if match:
            try:
                quiz_json = json.loads(match.group(0))
                save_message("ai", str(quiz_json))
                return jsonify({"quiz": quiz_json})
            except Exception as e:
                save_message("ai", reply)
                return jsonify({"reply": "Failed to parse quiz JSON."})
        else:
            save_message("ai", reply)
            return jsonify({"reply": "No quiz found in response."})
    else:
        save_message("ai", reply)
        return jsonify({"reply": reply})

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()

    # Only allow .txt, .pdf, .docx
    if ext not in [".txt", ".pdf", ".docx"]:
        return jsonify({"message": "Only .txt, .pdf, and .docx files are supported."}), 400

    try:
        if ext == ".txt":
            content = file.read().decode("utf-8")
        elif ext == ".pdf":
            reader = PdfReader(file)
            content = ""
            for page in reader.pages:
                content += page.extract_text() or ""
        elif ext == ".docx":
            doc = docx.Document(file)
            content = "\n".join([para.text for para in doc.paragraphs])
        else:
            return jsonify({"message": "Unsupported file type."}), 400
    except Exception as e:
        return jsonify({"message": f"Failed to process file: {str(e)}"}), 400

    if not content.strip():
        return jsonify({"message": "No extractable text found in the file."}), 400
    
    # Initialize knowledge graph
    kg = KnowledgeGraph()
    extracted_structures = []
    
    try:
        # Use semantic chunking for better knowledge extraction
        semantic_structured_chunks = semantic_chunks(content, similarity_threshold=0.65, max_length=900)
        
        for chunk in semantic_structured_chunks:
            structured = extract_structured_knowledge(chunk)
            if structured:
                # Save ONLY to Neo4j knowledge graph
                kg.create_concept_graph(structured)
                extracted_structures.append(structured)
                print(f"Created knowledge graph chunk with topic: {structured.get('Topic', 'Unknown')}")
                
    except Exception as e:
        print(f"Knowledge graph creation failed: {e}")
    finally:
        kg.close()

    # Regular RAG chunking (still needed for vector search)
    chunks = chunk_text(content, chunk_size=1000, overlap=200)
    conn = get_connection()
    cur = conn.cursor()
    for chunk in chunks:
        embedding = get_embedding(chunk)
        cur.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s::vector)",
            (chunk, f"[{','.join(str(x) for x in embedding)}]")
        )
    conn.commit()
    conn.close()
    return jsonify({
        "message": f"File uploaded: {len(chunks)} RAG chunks, {len(extracted_structures)} Neo4j knowledge concepts.",
        "structured_knowledge": extracted_structures
    })

@app.route("/score_quiz", methods=["POST"])
def score_quiz():
    data = request.json
    quiz = data.get("quiz", [])
    answers = data.get("answers", [])
    score = 0
    
    for q, user_answer in zip(quiz, answers):
        if not user_answer:  # Skip if no answer
            continue
            
        correct_answer = q.get("answer")
        
        # Simple letter-to-letter comparison
        def normalize_answer(answer):
            if not answer:
                return None
            
            # Extract just the letter part
            answer_str = str(answer).strip().upper()
            
            # If it's just a letter (A, B, C, D)
            if len(answer_str) == 1 and answer_str in ['A', 'B', 'C', 'D']:
                return answer_str
            
            # If it starts with a letter followed by delimiter
            if len(answer_str) > 1 and answer_str[0] in ['A', 'B', 'C', 'D']:
                return answer_str[0]
            
            return None

        user_letter = normalize_answer(user_answer)
        correct_letter = normalize_answer(correct_answer)
        
        print(f"Question: {q.get('question', '')[:50]}...")
        print(f"User answer: '{user_answer}' -> normalized: '{user_letter}'")
        print(f"Correct answer: '{correct_answer}' -> normalized: '{correct_letter}'")
        
        if user_letter and correct_letter and user_letter == correct_letter:
            score += 1
            print("✅ CORRECT")
        else:
            print("❌ INCORRECT")
        print("---")
                
    return jsonify({
        "score": score,
        "total": len(quiz),
        "answers": [q.get("answer") for q in quiz]
    })



@app.route("/learning_path", methods=["POST"])
def learning_path():
    data = request.json
    start = data.get("start_concept")
    end = data.get("end_concept")
    
    try:
        kg = KnowledgeGraph()
        path = kg.get_learning_path(start, end)
        kg.close()
        return jsonify({"path": path})
    except Exception as e:
        return jsonify({"error": f"Failed to find learning path: {str(e)}"})
    
@app.route("/graph_stats", methods=["GET"])
def graph_stats():
    try:
        kg = KnowledgeGraph()
        stats = kg.get_graph_stats()
        kg.close()
        return jsonify({"stats": stats})
    except Exception as e:
        return jsonify({"error": f"Failed to get graph stats: {str(e)}"})

@app.route("/related_topics", methods=["POST"])
def related_topics():
    data = request.json
    topic = data.get("topic")
    
    try:
        kg = KnowledgeGraph()
        related = kg.find_related_topics(topic)
        kg.close()
        return jsonify({"related_topics": related})
    except Exception as e:
        return jsonify({"error": f"Failed to find related topics: {str(e)}"})

@app.route("/test_neo4j", methods=["GET"])
def test_neo4j():
    try:
        kg = KnowledgeGraph()
        
        # Test basic functionality
        test_data = {
            "Topic": "Test Topic",
            "Subtopics": ["Test Subtopic"],
            "Definitions": ["Test Term: Test Definition"],
            "Terminology / Keywords": ["test", "keyword"]
        }
        
        kg.create_concept_graph(test_data)
        context = kg.get_concept_context(["test"])
        stats = kg.get_graph_stats()
        
        kg.close()
        
        return jsonify({
            "status": "success",
            "context_length": len(context),
            "stats": stats
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

def semantic_chunks(text, similarity_threshold=0.70, max_length=1200):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        return [text]

    # Step 1: embed all paragraphs
    embeddings = model.encode(paragraphs, convert_to_numpy=True)

    chunks = []
    current_chunk = paragraphs[0]
    prev_vec = embeddings[0]

    for i in range(1, len(paragraphs)):
        sim = np.dot(prev_vec, embeddings[i]) / (
            np.linalg.norm(prev_vec) * np.linalg.norm(embeddings[i])
        )

        # If meaning changes → new chunk
        if sim < similarity_threshold or len(current_chunk) > max_length:
            chunks.append(current_chunk.strip())
            current_chunk = paragraphs[i]
        else:
            current_chunk += "\n\n" + paragraphs[i]

        prev_vec = embeddings[i]

    chunks.append(current_chunk.strip())
    return chunks



if __name__ == "__main__":
    app.run(debug=True)
