import os
import json
import time
import uuid
from datetime import datetime
from supabase import create_client
from dotenv import load_dotenv
from openai import OpenAI
from test_engine import (
    QuestionGenerator,
    AnswerVerifier,
    ExplanationGenerator,
    ReportGenerator,
    TestWorkflowEngine,
    UserAnswer,
    logger,
    format_duration,
    setup_logger
)

# -------------------
# UTILITY: TOPIC EXTRACTION
# -------------------

# --- LLM-based topic extraction ---
def extract_topics_from_summary(summary, llm_client, max_topics=5):
    prompt = f"""
Extract 5 clear topic names from this chapter summary.

Rules:
- Only give short topic names
- No sentences
- No explanations
- Output as comma separated list

Summary:
{summary}
"""
    try:
        response = llm_client.chat.completions.create(
    model="inclusionai/ling-2.6-1t:free",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=256,
    temperature=0.2
)
        text = response.choices[0].message.content.strip()
        topics = [t.strip() for t in text.split(",") if t.strip()]
        return topics[:max_topics]
    except Exception as e:
        print("[Topic Extraction Error]", e)
        return []

def group_and_summarize_topics(raw_topics, llm_client, max_topics=3):
    if not raw_topics:
        return []
    prompt = f"""
Given this list of raw keywords from a student's test performance, group them into {max_topics} meaningful, broad conceptual test topics.

RAW KEYWORDS: {', '.join(raw_topics)}

Rules:
1. Output exactly {max_topics} topics max.
2. Ignore chemical formulas (like CO2, H2), generic words (water, heating), or single random words.
3. Group related keywords into broad concepts (e.g. "Acids and Bases", "Chemical Reactions", "Indian Independence").
4. Each topic must be 2-4 words max.
5. Provide ONLY a comma-separated list of the topics, no bullet points, no extra text.
"""
    try:
        response = llm_client.chat.completions.create(
    model="inclusionai/ling-2.6-1t:free",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=60,
    temperature=0.2
        )
        text = response.choices[0].message.content.strip()
        topics = [t.strip().lstrip('-').strip() for t in text.split(",") if t.strip()]
        return topics[:max_topics]
    except Exception as e:
        print("[Topic Summarization Error]", e)
        return list(raw_topics)[:max_topics]

# -------------------
# UTILITY: WEAK TOPICS
# -------------------
def get_weak_topics(student_id, chapter_id):
    try:
        res = supabase.table("weak_topics").select("topic,mistake_count").eq("student_id", student_id).eq("chapter_id", chapter_id).order("mistake_count", desc=True).limit(3).execute()
        return [x["topic"] for x in res.data]
    except Exception:
        return []

# -------------------
# TEST STATE
# -------------------
is_test_active = False
current_questions = []
current_answers = []
current_topics = []
current_difficulty = "easy"
current_chapter_id = None
current_student_id = None

# -------------------
# LOAD ENV
# -------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENROUTER_API_KEY = os.getenv("Openrouter_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# LLM (Llama)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Initialize components
question_generator = QuestionGenerator(client)
explanation_generator = ExplanationGenerator(client)
test_workflow = TestWorkflowEngine(client)


# -------------------
# FETCH DATA FROM SUPABASE
# -------------------

def get_standards():
    """Fetch all standards from database."""
    res = supabase.table("subjects").select("grade").execute()
    return sorted(list(set([x["grade"] for x in res.data])))


def get_subjects(grade):
    """Fetch subjects for a specific grade."""
    res = (
        supabase.table("subjects")
        .select("id,name")
        .eq("grade", grade)
        .execute()
    )
    return res.data


def get_chapters(subject_id):
    """Fetch chapters for a specific subject."""
    res = (
        supabase.table("chapters")
        .select("id,title")
        .eq("subject_id", subject_id)
        .execute()
    )
    return res.data


def get_chapter_summary(chapter_id):
    """Fetch chapter summary from database."""
    res = (
        supabase.table("chapters")
        .select("summary")
        .eq("id", chapter_id)
        .single()
        .execute()
    )
    return res.data["summary"]


# -------------------
# CHAT.PY TEST LOGIC MOVED HERE
# -------------------
# The following functions and logic were moved from chat.py:
# - PromptEnhancer.detect_query_type (test-related logic)
# - Test prompt building and evaluation logic
# - Test followup logic
# - Any test state management
#
# If you need to use test intent detection, test prompt building, or test evaluation, import or define them here.
#
# For example, you can add:
#
def detect_test_intent(query: str) -> bool:
    """Return True if the query is a test/quiz request (including Hinglish/Indian phrases)."""
    q = query.lower().strip()
    import re
    test_keywords = [
        "test", "quiz", "mcq", "question pucho", "exam",
        "test lo", "test me", "quiz le lo", "exam le lo", "question de", "mcq de", "practice test", "questions do", "sawal pucho", "sawal do", "question dedo", "questions dedo"
    ]
    return any(re.search(rf"(\\b|\s|^){re.escape(word)}(\\b|\s|$)", q) for word in test_keywords)

# Add any additional test prompt building or evaluation logic here as needed.


# -------------------
# INTERACTIVE TEST INTERFACE
# -------------------

def display_question(question, question_number):
    """Display a question with options."""
    print(f"\n{'='*70}")
    print(f"Question {question_number} of {question.id} | Difficulty: {question.difficulty.value.upper()}")
    print(f"{'='*70}")
    print(f"\n{question.question_text}\n")
    
    if question.question_type.value == "mcq":
        for idx, option in enumerate(question.options, 1):
            print(f"{idx}. {option}")
    elif question.question_type.value == "true_false":
        print("1. True")
        print("2. False")
    elif question.question_type.value == "short_answer":
        print("[Short Answer Question]")


def get_user_answer(question, question_number):
    """Get user's answer with timeout tracking."""
    display_question(question, question_number)
    
    start_time = time.time()
    
    while True:
        try:
            user_input = input(f"\nYour answer (or 'skip' to skip): ").strip()
            
            if user_input.lower() == 'skip':
                logger.info(f"Question {question.id} skipped")
                return None, time.time() - start_time
            
            if question.question_type.value == "mcq":
                try:
                    choice = int(user_input)
                    if 1 <= choice <= len(question.options):
                        answer = question.options[choice - 1]
                        logger.info(f"Question {question.id} answered with: {answer}")
                        return answer, time.time() - start_time
                    else:
                        print(f"❌ Please enter a number between 1 and {len(question.options)}")
                except ValueError:
                    print("❌ Please enter a valid number")
            
            elif question.question_type.value == "true_false":
                if user_input.lower() in ['1', '2', 'true', 'false', 't', 'f']:
                    mapping = {'1': 'True', '2': 'False', 'true': 'True', 'false': 'False', 't': 'True', 'f': 'False'}
                    answer = mapping.get(user_input.lower(), user_input)
                    logger.info(f"Question {question.id} answered with: {answer}")
                    return answer, time.time() - start_time
                else:
                    print("❌ Please enter 'true', 'false', 't', 'f', '1', or '2'")
            
            elif question.question_type.value == "short_answer":
                if user_input:
                    logger.info(f"Question {question.id} answered with: {user_input}")
                    return user_input, time.time() - start_time
                else:
                    print("❌ Please enter an answer")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Test interrupted by user")
            return None, time.time() - start_time


def evaluate_test(questions, user_answers):
    """Evaluate test and show feedback."""
    print(f"\n{'='*70}")
    print("📋 TEST EVALUATION")
    print(f"{'='*70}\n")
    
    verified_answers = []
    total_time = 0
    
    for idx, (question, user_answer_data) in enumerate(zip(questions, user_answers), 1):
        user_answer, time_taken = user_answer_data
        
        if user_answer is None:
            logger.info(f"Question {question.id} was skipped")
            continue
        
        total_time += time_taken
        
        # Verify answer
        is_correct = AnswerVerifier.verify_answer(
            user_answer,
            question.correct_answer,
            question.question_type
        )
        
        # Display result
        status = "✅ CORRECT" if is_correct else "❌ WRONG"
        print(f"\nQ{question.id}: {status}")
        print(f"Your Answer: {user_answer}")
        print(f"Correct Answer: {question.correct_answer}")
        print(f"Time Taken: {format_duration(time_taken)}")
        print(f"Explanation: {question.explanation}")
        
        # Generate detailed feedback for wrong answers
        if not is_correct:
            print("\n📝 Detailed Feedback:")
            try:
                feedback = explanation_generator.generate_wrong_answer_feedback(
                    question.question_text,
                    user_answer,
                    question.correct_answer,
                    question.explanation
                )
                print(f"   {feedback}\n")
                explanation = feedback
            except Exception as e:
                logger.warning(f"Failed to generate feedback: {e}")
                explanation = question.explanation
        else:
            explanation = ""
        
        # Create UserAnswer object
        verified_answer = UserAnswer(
            question_id=question.id,
            selected_answer=user_answer,
            is_correct=is_correct,
            time_taken=time_taken,
            explanation=explanation
        )
        verified_answers.append(verified_answer)
    
    return verified_answers, total_time


# -------------------
# EXPORT & REPORTING
# -------------------

def export_test_data(questions, filepath="test_questions.json"):
    """Export generated questions to JSON file."""
    question_generator.export_questions(questions, filepath)
    print(f"✅ Questions exported to {filepath}")


def create_test_report(student_name, chapter_title, questions, verified_answers, total_time):
    """Create and display test report."""
    report = ReportGenerator.generate_report(
        test_id=str(uuid.uuid4()),
        student_name=student_name,
        chapter_title=chapter_title,
        questions=questions,
        user_answers=verified_answers,
        total_time=total_time
    )
    
    ReportGenerator.print_report_summary(report)
    
    # Export report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"test_report_{student_name}_{timestamp}.json"
    ReportGenerator.export_report(report, report_file)
    print(f"📊 Report exported to {report_file}")
    
    return report


# -------------------
# MAIN FLOW
# -------------------

def run():
    """Main interactive test flow."""
    print("\n" + "="*70)
    print("🎓 CHILD-FRIENDLY AI MENTOR - TEST SYSTEM")
    print("="*70)
    

    global is_test_active, current_questions, current_answers, current_topics, current_difficulty, current_chapter_id, current_student_id
    try:
        # Get student name
        student_name = input("\nEnter your name: ").strip()
        if not student_name:
            student_name = "Anonymous"
        current_student_id = student_name

        # Select Standard
        print("\n📚 Select Standard:")
        standards = get_standards()
        for i, std in enumerate(standards, 1):
            print(f"{i}. {std}")
        std_choice = int(input("Enter choice: "))
        selected_std = standards[std_choice - 1]

        # Select Subject
        subjects = get_subjects(selected_std)
        print(f"\n📖 Select Subject:")
        for i, sub in enumerate(subjects, 1):
            print(f"{i}. {sub['name']}")
        sub_choice = int(input("Enter choice: "))
        selected_subject = subjects[sub_choice - 1]

        # Select Chapter
        chapters = get_chapters(selected_subject["id"])
        print(f"\n📝 Select Chapter:")
        for i, ch in enumerate(chapters, 1):
            print(f"{i}. {ch['title']}")
        ch_choice = int(input("Enter choice: "))
        selected_chapter = chapters[ch_choice - 1]
        current_chapter_id = selected_chapter["id"]

        print(f"\n✅ Selected Chapter: {selected_chapter['title']}")

        # Fetch chapter summary
        print("\n⏳ Fetching chapter summary...")
        summary = get_chapter_summary(selected_chapter["id"])

        # Topic selection
        print("\n🔎 Extracting topics...")

        summary_topics = extract_topics_from_summary(summary, client)
        weak_topics = get_weak_topics(current_student_id, current_chapter_id)
        suggested_topics = weak_topics + [t for t in summary_topics if t not in weak_topics]
        print("\nSuggested topics:")
        for i, t in enumerate(suggested_topics, 1):
            print(f"  {i}. {t}")
        topic_choice = input("Pick a topic number or press Enter for all: ").strip()
        if topic_choice.isdigit() and 1 <= int(topic_choice) <= len(suggested_topics):
            current_topics = [suggested_topics[int(topic_choice)-1]]
        else:
            current_topics = suggested_topics[:3] if suggested_topics else summary_topics[:3]

        # Difficulty selection
        print("\nSelect difficulty:")
        print("  1. Easy\n  2. Medium\n  3. Hard")
        diff_map = {"1": "easy", "2": "medium", "3": "hard"}
        diff_choice = input("Enter choice (1-3): ").strip()
        current_difficulty = diff_map.get(diff_choice, "easy")


        # Generate test (10 MCQs only)
        print(f"\n⏳ Generating 10 MCQ questions on topics: {', '.join(current_topics)} | Difficulty: {current_difficulty}")
        is_test_active = True
        current_questions = question_generator.generate_test_json(
            summary + "\nTopics: " + ", ".join(current_topics) + f"\nDifficulty: {current_difficulty}\nOnly MCQ questions. No True/False.",
            num_questions=10
        )
        # Filter for MCQ only
        current_questions = [q for q in current_questions if q.question_type.value == "mcq"][:10]

        print(f"\n{'='*70}")
        print(f"🎯 TEST READY - {len(current_questions)} MCQ QUESTIONS")
        print(f"{'='*70}")
        print("Instructions:")
        print("- Answer each question (type the option letter)")
        print("- Type 'skip' to skip a question")
        print("- Type 'quit' to exit test mode early")
        print(f"{'='*70}\n")

        user_answers_with_time = []
        for idx, question in enumerate(current_questions, 1):
            if not is_test_active:
                break
            ans, t = get_user_answer(question, idx)
            if ans == "quit":
                print("Exiting test mode.")
                is_test_active = False
                break
            user_answers_with_time.append((ans, t))

        total_test_time = sum(t for _, t in user_answers_with_time)

        # Fast evaluation (no LLM)
        print(f"\n{'='*70}")
        print("📋 TEST EVALUATION")
        print(f"{'='*70}\n")
        correct = 0
        wrong = 0
        mistakes = {}
        strong_topics = set()
        weak_topics_eval = set()
        for idx, (question, (user_answer, _)) in enumerate(zip(current_questions, user_answers_with_time), 1):
            if user_answer is None or user_answer == "skip":
                print(f"Q{question.id}: Skipped")
                continue
            ca = question.correct_answer.strip().lower()
            ua = str(user_answer).strip().lower()
            is_correct = (ua == ca or (question.question_type.value == "true_false" and ua in [ca, ca[0]]))
            if is_correct:
                print(f"Q{question.id}: ✅ Correct")
                correct += 1
                for k in question.keywords:
                    strong_topics.add(k)
            else:
                print(f"Q{question.id}: ❌ Wrong | Correct: {question.correct_answer}\n   Explanation: {question.explanation}\n   Topic: {', '.join(question.keywords)}")
                wrong += 1
                for k in question.keywords:
                    mistakes[k] = mistakes.get(k, 0) + 1
                    weak_topics_eval.add(k)

        print(f"\nScore: {correct}/{len(current_questions)}")
        print(f"Time Taken: {format_duration(total_test_time)}")

        # Confidence loop: strong/weak topics
        print()
        
        # Sort and prioritize raw topics
        raw_strong = list(strong_topics)
        raw_weak = [t[0] for t in sorted(mistakes.items(), key=lambda x: (-x[1], x[0]))]
        
        # Filter and summarize with LLM
        final_strong = group_and_summarize_topics(raw_strong, client, max_topics=3) if raw_strong else []
        final_weak = group_and_summarize_topics(raw_weak, client, max_topics=3) if raw_weak else []
        
        if final_strong:
            print("You’re improving in:")
            for t in final_strong:
                print(f"- {t}")
                
        if final_weak:
            print("\nYou need help in:")
            for t in final_weak:
                print(f"- {t}")
            
            top_weak_topic = final_weak[0]
            print(f"\n👉 Let’s fix {top_weak_topic} together\n")

        # Generate and export test report
        verified_answers = []
        for idx, (question, (user_answer, time_taken)) in enumerate(zip(current_questions, user_answers_with_time), 1):
            ca = question.correct_answer.strip().lower()
            ua = str(user_answer).strip().lower() if user_answer else ""
            is_correct = (ua == ca or (question.question_type.value == "true_false" and ua in [ca, ca[0]]))
            explanation = question.explanation if not is_correct else ""
            verified_answers.append(UserAnswer(
                question_id=question.id,
                selected_answer=user_answer if user_answer else "",
                is_correct=is_correct,
                time_taken=time_taken,
                explanation=explanation
            ))
        report = ReportGenerator.generate_report(
            test_id=str(uuid.uuid4()),
            student_name=student_name,
            chapter_title=selected_chapter['title'],
            questions=current_questions,
            user_answers=verified_answers,
            total_time=total_test_time
        )
        ReportGenerator.print_report_summary(report)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"test_report_{student_name}_{timestamp}.json"
        ReportGenerator.export_report(report, report_file)
        print(f"📊 Report exported to {report_file}\n")

        # Smart feedback and update weak_topics in DB
        if mistakes:
            sorted_weak = sorted(mistakes.items(), key=lambda x: -x[1])[:3]
            weak_list = [w[0] for w in sorted_weak]
            # Update Supabase weak_topics
            for w in weak_list:
                try:
                    prev = supabase.table("weak_topics").select("mistake_count").eq("student_id", current_student_id).eq("chapter_id", current_chapter_id).eq("topic", w).execute()
                    if prev.data:
                        supabase.table("weak_topics").update({"mistake_count": prev.data[0]["mistake_count"]+1}).eq("student_id", current_student_id).eq("chapter_id", current_chapter_id).eq("topic", w).execute()
                    else:
                        supabase.table("weak_topics").insert({"student_id": current_student_id, "chapter_id": current_chapter_id, "topic": w, "mistake_count": 1}).execute()
                except Exception:
                    pass

        # Next best action engine
        print()
        score_pct = (correct / len(current_questions)) * 100 if current_questions else 0
        if score_pct < 50:
            print("Let's review your weak topics together and try some easy practice questions next!")
        elif score_pct < 80:
            print("Good effort! Want to retry this test or revise the explanations?")
        else:
            print("Awesome! Ready for a harder test or the next topic?")

        print(f"\n{'='*70}\n")
        is_test_active = False

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        is_test_active = False


if __name__ == "__main__":
    run()