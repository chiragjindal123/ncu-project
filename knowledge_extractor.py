import requests, json
from rag_utils import get_connection
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = os.getenv("GEMINI_URL")

EXTRACTION_PROMPT = """
You are an AI specialized in converting lecture slides or documents 
into structured academic knowledge.

Extract the following fields from the text:

- Topic
- Subtopics (list)
- Definitions (list)
- Formulas (list)
- Terminology / Keywords (list)
- Learning Outcomes (list)
- Diagram Descriptions (list)

Return ONLY valid JSON with these keys.
"""

def extract_structured_knowledge(text):
    payload = {
        "contents": [{
            "parts": [{"text": EXTRACTION_PROMPT + "\n\nTEXT:\n" + text[:4000]}]  # Limit text size
        }]
    }
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }

    try:
        r = requests.post(GEMINI_URL, headers=headers, data=json.dumps(payload))
        out = r.json()
        
        # print("API Response:", out)  # Debug: see full response
        
        if "candidates" in out and out["candidates"]:
            response_text = out["candidates"][0]["content"]["parts"][0]["text"]
            # print("Response text:", response_text)  # Debug: see response text
            
            # Try to extract JSON from response
            import re
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                # print("Extracted JSON:", json_str)  # Debug: see extracted JSON
                return json.loads(json_str)
            else:
                print("No JSON found in response")
                return None
        elif "error" in out:
            print("API Error:", out["error"])
            return None
        else:
            print("Unexpected response structure:", out)
            return None
            
    except json.JSONDecodeError as e:
        print("JSON parsing failed:", e)
        return None
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        return None
    except Exception as e:
        print("Unexpected error:", e)
        return None


def save_to_db(structure):
    if not structure:
        return
        
    conn = get_connection()
    cur = conn.cursor()

    try:
        topic = structure.get("Topic", "Unknown Topic")
        subtopics = structure.get("Subtopics", [])
        
        if not subtopics:
            subtopics = [topic]

        for sub in subtopics:
            cur.execute("""
                INSERT INTO course_knowledge
                (topic, subtopic, definition, formula, keywords, learning_outcome, diagram_caption, raw_text)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s);
            """, (
                topic,
                sub,
                "\n".join(structure.get("Definitions", [])),
                "\n".join(structure.get("Formulas", [])),
                "\n".join(structure.get("Terminology / Keywords", [])),  # Fixed: use correct key
                "\n".join(structure.get("Learning Outcomes", [])),       # Fixed: use correct key
                "\n".join(structure.get("Diagram Descriptions", [])),    # Fixed: use correct key
                ""  # raw_text
            ))
        
        conn.commit()
        print(f"Successfully saved {len(subtopics)} knowledge entries")
        
    except Exception as e:
        print("Database save failed:", e)
        conn.rollback()
    finally:
        conn.close()