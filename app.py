from flask import Flask, render_template, request, jsonify
import requests
import subprocess
import psycopg2
import json
import re
from rag_utils import get_context, get_embedding, save_message, get_connection, chunk_text
from knowledge_extractor import extract_structured_knowledge, save_to_db
import os
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import docx
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np

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

    
    # Get structured context (UPDATE this section in your chat route)
    structured_context = ""
    if use_structure:
        conn = get_connection()
        cur = conn.cursor()
        
        # Debug: Check what we're searching for
        print(f"Searching structured knowledge for: '{user_input}'")
        
        # Split the user input into individual words for better searching
        search_words = user_input.lower().split()
        search_conditions = []
        search_params = []
        
        for word in search_words:
            if len(word) > 2:  # Only search for words longer than 2 characters
                search_conditions.extend([
                    "topic ILIKE %s",
                    "subtopic ILIKE %s", 
                    "keywords ILIKE %s",
                    "definition ILIKE %s"
                ])
                search_params.extend([f"%{word}%", f"%{word}%", f"%{word}%", f"%{word}%"])
        
        if search_conditions:
            query = f"""
                SELECT DISTINCT topic, subtopic, definition, formula, keywords 
                FROM course_knowledge 
                WHERE {' OR '.join(search_conditions)}
                LIMIT 10;
            """
            cur.execute(query, search_params)
        else:
            # Fallback: get some recent entries if no good search terms
            cur.execute("SELECT topic, subtopic, definition, formula, keywords FROM course_knowledge LIMIT 5;")
        
        rows = cur.fetchall()
        conn.close()

        print(f"Found {len(rows)} structured knowledge entries")  # Debug
        
        # In your structured context section, update this part:
        if rows:
            structured_context = "STRUCTURED ACADEMIC KNOWLEDGE:\n\n"
            for r in rows:
                if r[0]:  # if topic exists
                    structured_context += f"**Topic**: {r[0]}\n"
                    structured_context += f"**Subtopic**: {r[1]}\n" 
                    structured_context += f"**Definition**: {r[2]}\n"
                    if r[3]:  # if formula exists
                        structured_context += f"**Formula**: {r[3]}\n"
                    structured_context += f"**Keywords**: {r[4]}\n"
                    structured_context += "---\n\n"
        
        print(f"Structured context: {structured_context[:200]}...")  # Debug first 200 chars
        
    # --- Prompt Engineering ---
    if task == "review":
        task_prompt = f"Review the following material and explain it simply:\n{user_input}"
    elif task == "quiz":
        task_prompt = (
            "Generate 5 multiple-choice quiz questions in the following JSON format:\n"
            '[{"question": "...", "options": ["A", "B", "C", "D"], "answer": "A"}, ...]\n'
            f"Material:\n{user_input}\n"
            "Do not include explanations. Only output valid JSON."
        )
    elif task == "practice":
        task_prompt = (
            "Give me a practical implementation exercise (with a brief solution):\n"
            f"{user_input}"
        )
    else:
        task_prompt = user_input
        
    if use_structure and use_rag:
        system_instruction = "You are an AI Teaching Assistant. Use both the structured academic knowledge and RAG context to provide a comprehensive, well-formatted response with headings, bullet points, and academic structure."
    elif use_structure:
        system_instruction = "You are an AI Teaching Assistant. Use the structured academic knowledge to provide an organized, pedagogical response with clear definitions, topics, and academic formatting."
    elif use_rag:
        system_instruction = "You are an AI Teaching Assistant. Use the provided context to give a detailed, well-formatted response with bullet points, bold headings, and clear organization."
    else:
        system_instruction = "You are an AI Teaching Assistant. Provide a helpful response based on your knowledge."


    # final_prompt = f"Context:\n{rag_context}\n\nTask:\n{task_prompt}"
    
    # Combined prompt with structured knowledge
    # final_prompt = f"Structured Knowledge:\n{structured_context}\n\nRAG Context:\n{rag_context}\n\nTask:\n{task_prompt}"
    final_prompt = f"{system_instruction}\n\nStructured Knowledge:\n{structured_context}\n\nRAG Context:\n{rag_context}\n\nTask:\n{task_prompt}"

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
    
     # Semantic chunking for structured extraction
    semantic_structured_chunks = semantic_chunks(content, similarity_threshold=0.65, max_length=900)
    extracted_structures = []
    for chunk in semantic_structured_chunks:
        structured = extract_structured_knowledge(chunk)
        if structured:
            save_to_db(structured)
            extracted_structures.append(structured)


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
        "message": f"File uploaded and embedded in {len(chunks)} RAG chunks and {len(extracted_structures)} structured chunks.",
        "structured_knowledge": extracted_structures  # Send extracted knowledge to frontend
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
        
        # Convert full answers to just their option letter
        def get_letter(answer):
            if not answer:
                return None
            # If it's just a letter, return it
            if len(answer) == 1:
                return answer.upper()
            # If it starts with a letter and a dot/space/period
            if len(answer) > 1 and answer[0].isalpha() and answer[1] in '. ':
                return answer[0].upper()
            # If it's a full answer, check for exact match or contained text
            for i, opt in enumerate(q.get("options", [])):
                if answer.lower().strip() == opt.lower().strip():
                    return chr(65 + i)  # Convert 0->A, 1->B, etc.
            return None

        user_letter = get_letter(user_answer)
        correct_letter = get_letter(correct_answer)
        
        if user_letter and correct_letter and user_letter == correct_letter:
            score += 1
                
    return jsonify({
        "score": score,
        "total": len(quiz),
        "answers": [q.get("answer") for q in quiz]
    })






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
