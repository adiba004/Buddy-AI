import os
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
import json
import re
from typing import List, Dict, Tuple, TypedDict, Any
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

# -------------------
# CONFIGURATION
# -------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENROUTER_API_KEY = os.getenv("Openrouter_API_KEY") 

# -------------------
# CONSTANTS
# -------------------
MAX_HISTORY_TURNS = 5
MAX_CONTEXT_LENGTH = 3000
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "inclusionai/ling-2.6-1t:free"

# -------------------
# INIT CLIENTS
# -------------------
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Failed to initialize Supabase: {e}")
    supabase = None

try:
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
except Exception as e:
    print(f"Failed to initialize SentenceTransformer: {e}")
    embed_model = None

try:
    from openai import OpenAI
    llm_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
except Exception as e:
    print(f"Failed to initialize OpenRouter client: {e}")
    llm_client = None

# -------------------
# CONVERSATION MEMORY
# -------------------
class ConversationMemory:
    def __init__(self, max_turns: int = MAX_HISTORY_TURNS):
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []

    def add_turn(self, query: str, response: str) -> None:
        self.history.append({
            "query": query,
            "response": response
        })
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def get_context_string(self) -> str:
        if not self.history:
            return ""
        parts = ["Previous conversation:"]
        for turn in self.history[-3:]:
            parts.append(f"Student: {turn['query']}")
            parts.append(f"Buddy AI: {turn['response'][:200]}...")
        return "\n".join(parts)

    def clear(self) -> None:
        self.history = []


# -------------------
# PROMPT ENHANCER
# -------------------
class PromptEnhancer:
    @staticmethod
    def detect_query_type(query: str) -> str:
        q = query.lower().strip()
        import re
        if any(w in q for w in ["evaluate", "check my answer", "my answer is", "is this correct", "am i right", "is it right"]):
            return "evaluate"
        elif any(w in q for w in ["summary", "summarize", "recap", "overview", "tl;dr"]):
            return "summary"
        confusion_keywords = ["samaj ni", "samaj nahi", "samjha nahi", "samjh ni", "kuch ni samaj", "confuse", "not understand", "don't get", "explain", "difficult", "problem", "help"]
        if any(w in q for w in confusion_keywords):
            return "doubt"
        return "doubt"

    @staticmethod
    def get_child_friendly_instructions() -> str:
        return """You are Buddy AI, a friendly and patient tutor for Class 9 students.

Rules:
1. Use simple, age-appropriate language
2. Be encouraging and positive
3. Use daily life examples
4. Break down complex concepts
5. Keep responses concise but complete
6. Use emojis sparingly
7. Never be harsh or condescending"""

    @staticmethod
    def enhance_query(query: str, chapter_title: str, grade: str, subject: str, memory: 'ConversationMemory') -> str:
        prompt = f"""You are a query enhancer for an AI tutor.
Original Query: "{query}"
Chapter: "{chapter_title}"
Student's Grade: {grade}
Subject: {subject}
Conversation History: {memory.get_context_string()}

Task:
- Expand short/unclear queries into a clear, structured instruction.
- Add intent (explain, summarize, test, compare, etc.).
- Add student level context (Class {grade}).
- Use '{grade}' as the grade in all instructions and examples.
- Keep it simple and structured.
- If it is already fully clear, output it exactly as is.

Examples:
"ye kya hai" -> "Explain this concept in simple terms for a class {grade} student with an example."
"test lo" -> "Generate a short test with MCQs and answers for this chapter for a class {grade} student."
"samjha nahi" -> "Explain the previous concept again in a simpler way with an easy example for a class {grade} student."

DO NOT answer the query! Output ONLY the enhanced query text on a single line without quotes.
"""
        ans = generate_answer(prompt, temperature=0.3).strip().replace('"', '')
        if "Sorry, I encountered an error" in ans or "Mock model offline" in ans:
            return query
        return ans


# -------------------
# LLM CALL
# -------------------
def generate_answer(prompt: str, temperature: float = 0.7) -> str:
    try:
        if not llm_client:
            return "Mock model offline. Couldn't generate an answer."
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=temperature,
            extra_headers={
                "HTTP-Referer": "https://buddy-ai.app",   # optional but recommended by OpenRouter
                "X-Title": "Buddy AI",
            }
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}. Please try again!"


# -------------------
# FETCH DATA
# -------------------
def get_standards() -> List[str]:
    try:
        res = supabase.table("subjects").select("grade").execute()
        return sorted(list(set([x["grade"] for x in res.data])))
    except Exception as e:
        print(f"Error fetching standards: {e}")
        return []

def get_subjects(grade: str) -> List[Dict]:
    try:
        res = supabase.table("subjects").select("id,name").eq("grade", grade).execute()
        return res.data
    except Exception as e:
        print(f"Error fetching subjects: {e}")
        return []

def get_chapters(subject_id: str) -> List[Dict]:
    try:
        # Added 'summary' to our fetched properties.
        res = supabase.table("chapters").select("id,title,content,summary").eq("subject_id", subject_id).execute()
        return res.data
    except Exception as e:
        print(f"Error fetching chapters: {e}")
        return []


# -------------------
# VECTOR SEARCH
# -------------------
def get_relevant_chunks(query: str, chapter_id: str, k: int = 3) -> List[str]:
    try:
        if not embed_model:
            return []
        query_embedding = embed_model.encode(query).tolist()
        response = supabase.rpc(
            "match_chapters_filter",
            {"query_embedding": query_embedding, "match_count": k, "input_chapter_id": chapter_id}
        ).execute()
        return [x["content_chunk"] for x in response.data] if response.data else []
    except Exception as e:
        print(f"Error in vector search: {e}")
        return []


# -------------------
# LANGGRAPH STATE
# -------------------
class AgentState(TypedDict):
    query: str
    chapter: Dict
    memory: ConversationMemory
    intent: str
    context_text: str
    prompt: str
    response: str
    user_selections: Dict
    current_test: str


# -------------------
# LANGGRAPH NODES
# -------------------
def intent_detection_node(state: AgentState) -> Dict:
    query = state["query"]
    intent = PromptEnhancer.detect_query_type(query)
    return {"intent": intent}


def context_retrieval_node(state: AgentState) -> Dict:
    query = state["query"]
    chapter = state["chapter"]
    intent = state["intent"]
    
    chapter_summary_text = chapter.get("summary")
    if not chapter_summary_text:
        # Fallback to the first MAX_CONTEXT_LENGTH characters if schema missing summary
        chapter_summary_text = chapter.get("content", "")[:MAX_CONTEXT_LENGTH]
    
    if intent in ["summary", "test"]:
        return {"context_text": chapter_summary_text[:MAX_CONTEXT_LENGTH]}
    elif intent == "doubt":
        # Limit to top 3 chunks for RAG
        chunks = get_relevant_chunks(query, chapter["id"], k=3)
        joined_chunks = "\n\n".join(chunks) if chunks else ""
        return {"context_text": joined_chunks}
    else:
        # Evaluate
        return {"context_text": chapter_summary_text[:MAX_CONTEXT_LENGTH]}


def prompt_building_node(state: AgentState) -> Dict:
    query = state["query"]
    intent = state["intent"]
    context_text = state["context_text"]
    memory = state["memory"]
    chapter = state["chapter"]
    
    history_context = memory.get_context_string()

    if intent == "doubt" and not context_text.strip():
        prompt = f"""You are Buddy AI. The student asked an out-of-context question.

Student Question: {query}
Current Chapter: {chapter.get('title', '')}

Task:
1. Answer their question VERY BRIEFLY (max 2-3 lines). Keep it simple.
2. Immediately after your brief answer, gently remind them that we are currently studying {chapter.get('title', '')}.
3. Ask if they want to continue with the current chapter or connect this concept to it.
4. Keep the tone warm and friendly 😊

DO NOT output any internal monologues or generic follow-ups.
"""
        return {"prompt": prompt}


    prompt = f"""You are Buddy AI, a warm and friendly tutor for Indian school students.\n\nClass: {{standard}}\nSubject: {{subject}}\nChapter: {{chapter_title}}\n\n════════════════════════════════════════════════\nDETECTED INTENT: {{intent}}\n════════════════════════════════════════════════\n\nYOUR ONLY JOB BASED ON INTENT:\n\nIf intent is \"doubt\":\n→ The student is confused. Explain in simple language with one small analogy.\n→ Keep it under 6 lines.\n→ ⛔ ABSOLUTELY NO quiz, MCQ, or True/False. ZERO questions.\n\nIf intent is \"summary\":\n→ Give a clean bullet-point summary. Max 6 bullets. Key points only.\n→ ⛔ ABSOLUTELY NO quiz, MCQ, or True/False. ZERO questions.\n\nIf intent is \"evaluate\":\n→ The student shared an answer. Check it and give kind feedback.\n→ ⛔ ABSOLUTELY NO quiz, MCQ, or True/False. ZERO questions.\n\nIf intent is \"test\":\n→ ONLY in this case you may give a quiz. Max 3 questions. Then STOP and wait.\n\n════════════════════════════════════════════════\n⛔ HARD RULE — READ THIS BEFORE RESPONDING:\nIf DETECTED INTENT is anything other than \"test\",\nyou are BANNED from writing any quiz, MCQ, True/False,\nor numbered question list. Even if the content feels\nrelated. Even if it seems helpful. DO NOT DO IT.\n════════════════════════════════════════════════\n\nHandle Hinglish naturally:\n- \"samaj nahi / samaj ni / kuch ni samaj\" = confused, EXPLANATION needed\n- \"eg / eg do / eg dedo / give eg\" = wants an EXAMPLE\n- \"batao / bata\" = explain to me\n- \"quiz / test / MCQ / question pucho\" = ONLY these mean they want a test\n\nConversation History:\n{history_context}\n\nChapter Content:\n{context_text}\n\nStudent's Message: {query}\n\nNow respond based on the intent \"{intent}\" above.\nRemember: ⛔ NO quiz unless intent is exactly \"test\".\n"""
    # Fill in the variables for the prompt
    prompt = prompt.replace("{standard}", str(state['user_selections'].get('standard', '9')))
    prompt = prompt.replace("{subject}", str(state['user_selections'].get('subject', '')))
    prompt = prompt.replace("{chapter_title}", str(chapter.get('title', '')))
    prompt = prompt.replace("{intent}", intent)
    prompt = prompt.replace("{history_context}", history_context)
    prompt = prompt.replace("{context_text}", context_text)
    prompt = prompt.replace("{query}", query)
    return {"prompt": prompt}


def llm_call_node(state: AgentState) -> Dict:
    prompt = state["prompt"]
    intent = state["intent"]
    raw_response = generate_answer(prompt)
    
    return {"response": raw_response}


def memory_update_node(state: AgentState) -> Dict:
    query = state["query"]
    response = state["response"]
    memory = state["memory"]
    memory.add_turn(query, response)
    
    return {"memory": memory}


# -------------------
# BUILD LANGGRAPH GRAPH
# -------------------
def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("intent_detection", intent_detection_node)
    workflow.add_node("context_retrieval", context_retrieval_node)
    workflow.add_node("prompt_building", prompt_building_node)
    workflow.add_node("llm_call", llm_call_node)
    workflow.add_node("memory_update", memory_update_node)
    
    workflow.set_entry_point("intent_detection")
    
    workflow.add_edge("intent_detection", "context_retrieval")
    workflow.add_edge("context_retrieval", "prompt_building")
    workflow.add_edge("prompt_building", "llm_call")
    workflow.add_edge("llm_call", "memory_update")
    workflow.add_edge("memory_update", END)
    
    return workflow.compile()


def evaluate_test_answers(student_answers: str, test_content: str, chapter_title: str) -> str:
    prompt = f"""You are Buddy AI. Evaluate the test.

Stored JSON Test (with answers):
{test_content}

Student's Answers:
{student_answers}

Task:
Compare answers.
1. Output 'INVALID_FORMAT_RETRY' if completely unreadable.
2. Otherwise, format exactly like this:

Score: [X]/5 👍

Q1: Correct
Q2: Incorrect → Correct answer: [A/B/C/D]
Explanation: [Very simple reason]

Weak Topics: [Topic1, Topic2]
"""
    return generate_answer(prompt)

def generate_followup(intent: str, score: int = None, weak_topics: list = None) -> str:
    # Test followup removed
    if intent == "summary":
        return "Want me to test you on this or explain any part more simply?"
    if intent == "doubt":
        return "Want a quick example to understand this better?"
    return "What would you like to explore next?"

# -------------------
# MAIN RUN LOOP
# -------------------
def run():
    print(f"\n{'='*50}")
    print(f"{'Buddy AI (MVP)' :^50}")
    print(f"{'='*50}\n")
    
    if not supabase:
        print("Missing Supabase configuration. Ensure standard .env keys are attached.")
        return

    memory = ConversationMemory()

    print("Available Standards:")
    standards = get_standards()
    if not standards:
        print("No standards found. Please check your DB connection.")
        return
    for i, std in enumerate(standards, 1):
        print(f"  {i}. {std}")
    try:
        std_choice = int(input("\nEnter choice number: ")) - 1
        if std_choice < 0 or std_choice >= len(standards):
            print("Invalid choice!"); return
    except ValueError:
        print("Enter a valid number!"); return
    selected_std = standards[std_choice]

    subjects = get_subjects(selected_std)
    if not subjects:
        print("No subjects found."); return
    print("\nAvailable Subjects:")
    for i, sub in enumerate(subjects, 1):
        print(f"  {i}. {sub['name']}")
    try:
        sub_choice = int(input("\nEnter choice number: ")) - 1
        if sub_choice < 0 or sub_choice >= len(subjects):
            print("Invalid choice!"); return
    except ValueError:
        print("Enter a valid number!"); return
    selected_subject = subjects[sub_choice]

    chapters = get_chapters(selected_subject["id"])
    if not chapters:
        print("No chapters found."); return
    print("\nAvailable Chapters:")
    for i, ch in enumerate(chapters, 1):
        print(f"  {i}. {ch['title']}")
    try:
        ch_choice = int(input("\nEnter choice number: ")) - 1
        if ch_choice < 0 or ch_choice >= len(chapters):
            print("Invalid choice!"); return
    except ValueError:
        print("Enter a valid number!"); return
    selected_chapter = chapters[ch_choice]

    user_selections = {
        "standard": selected_std,
        "subject": selected_subject["name"],
        "chapter_title": selected_chapter["title"],
        "chapter_id": selected_chapter["id"]
    }

    app = build_graph()
    state = {
        "query": "",
        "chapter": selected_chapter,
        "memory": memory,
        "intent": "",
        "context_text": "",
        "prompt": "",
        "response": "",
        "user_selections": user_selections,
        "current_test": ""
    }

    print(f"\nSelected: {selected_std} > {selected_subject['name']} > {selected_chapter['title']}")
    print("\nCommands: /exit | /clear")

    use_enhanced_query = True

    while True:
        try:
            query = input("\nYou: ").strip()
            if not query:
                continue
            if query.lower() == "/exit":
                print("\nThanks for learning with Buddy AI! Keep studying!\n")
                break
            elif query.lower() == "/clear":
                memory.clear()
                print("Conversation history cleared!\n")
                continue
            elif query.lower() == "/raw":
                use_enhanced_query = False
                print("\n⚙️ Testing Mode: Enhancer DISABLED. Using raw queries.\n")
                continue
            elif query.lower() == "/enhanced":
                use_enhanced_query = True
                print("\n⚙️ Testing Mode: Enhancer ENABLED. Using expanded queries.\n")
                continue

            q_lower = query.lower()
            control_signals = [
                "yes", "ok", "okay", "haan", "hmm", "yep", "yeah",
                "quick", "very quick", "short", "simple", "brief",
                "eg", "example", "give example", "samjha nahi"
            ]

            is_control = False
            if q_lower in control_signals:
                is_control = True
            elif len(query.split()) < 4:
                is_control = True

            if is_control:
                enhanced_query = query
                print(f"[DEBUG] Control Signal '{query}' bypassed enhancer.")
            else:
                enhanced_query = PromptEnhancer.enhance_query(
                    query,
                    selected_chapter['title'],
                    selected_std,
                    selected_subject['name'],
                    memory
                )
                print(f"[DEBUG] Original: {query}")
                print(f"[DEBUG] Enhanced: {enhanced_query}")

            if use_enhanced_query:
                query = enhanced_query

            state["query"] = query
            result = app.invoke(state)
            state.update(result)

            print(f"\nBuddy AI:\n{result['response']}\n")

            intent = result.get("intent", "doubt")
            followup = generate_followup(intent)
            print("👉 " + followup)
        except KeyboardInterrupt:
            print("\n\nGoodbye! Stay curious!\n")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    run()
