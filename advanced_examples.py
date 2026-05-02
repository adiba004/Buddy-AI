"""
Advanced Usage Examples & Best Practices
Demonstrates real-world scenarios and advanced patterns
"""

import json
from typing import List
from datetime import datetime
from test_engine import (
    QuestionGenerator,
    AnswerVerifier,
    ExplanationGenerator,
    ReportGenerator,
    TestWorkflowEngine,
    UserAnswer,
    Question,
    DifficultyLevel,
    QuestionType,
    logger,
    DEFAULT_CONFIG
)
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=HF_TOKEN
)


# =====================================================
# EXAMPLE 1: Batch Generate Questions for Multiple Chapters
# =====================================================

def batch_generate_questions(chapters_data: List[dict], output_dir: str = "batch_output"):
    """
    Generate questions for multiple chapters at once.
    
    Example:
        chapters_data = [
            {"id": 1, "title": "Photosynthesis", "summary": "..."},
            {"id": 2, "title": "Respiration", "summary": "..."}
        ]
        batch_generate_questions(chapters_data)
    """
    os.makedirs(output_dir, exist_ok=True)
    gen = QuestionGenerator(client)
    
    results = []
    for chapter in chapters_data:
        try:
            logger.info(f"Generating questions for: {chapter['title']}")
            
            questions = gen.generate_test_json(chapter['summary'], num_questions=10)
            
            # Export
            filename = f"{output_dir}/questions_{chapter['id']}_{chapter['title']}.json"
            gen.export_questions(questions, filename)
            
            results.append({
                "chapter_id": chapter['id'],
                "chapter_title": chapter['title'],
                "question_count": len(questions),
                "export_file": filename,
                "status": "success"
            })
            
            logger.info(f"✅ Generated {len(questions)} questions for {chapter['title']}")
        
        except Exception as e:
            logger.error(f"❌ Failed to generate for {chapter['title']}: {e}")
            results.append({
                "chapter_id": chapter['id'],
                "chapter_title": chapter['title'],
                "status": "failed",
                "error": str(e)
            })
    
    # Export results summary
    summary_file = f"{output_dir}/batch_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Batch processing complete. Summary: {summary_file}")
    return results


# =====================================================
# EXAMPLE 2: Load Questions from File & Run Offline Test
# =====================================================

def run_offline_test_from_file(questions_file: str, student_name: str = "Student"):
    """
    Load pre-generated questions and run test without regenerating.
    Useful for running same test multiple times.
    
    Example:
        run_offline_test_from_file("questions_photosynthesis.json", "John Doe")
    """
    logger.info(f"Loading questions from {questions_file}")
    
    # Load questions
    with open(questions_file, 'r') as f:
        data = json.load(f)
    
    # Parse into Question objects
    questions = []
    for q_data in data['questions']:
        question = Question(
            id=q_data['id'],
            question_text=q_data['question_text'],
            question_type=QuestionType(q_data['question_type']),
            options=q_data['options'],
            correct_answer=q_data['correct_answer'],
            explanation=q_data['explanation'],
            difficulty=DifficultyLevel(q_data['difficulty']),
            keywords=q_data.get('keywords', [])
        )
        questions.append(question)
    
    print(f"\n📋 Loaded {len(questions)} questions")
    
    # Simulate taking test
    verified_answers = []
    for idx, question in enumerate(questions, 1):
        print(f"\nQ{idx}: {question.question_text}")
        print(f"Difficulty: {question.difficulty.value}")
        
        # Mock answer (in real scenario, get from user)
        user_answer = question.options[0]  # For demo
        
        is_correct = AnswerVerifier.verify_answer(
            user_answer,
            question.correct_answer,
            question.question_type
        )
        
        ua = UserAnswer(
            question_id=question.id,
            selected_answer=user_answer,
            is_correct=is_correct,
            time_taken=30.0
        )
        verified_answers.append(ua)
    
    # Generate report
    report = ReportGenerator.generate_report(
        test_id=f"offline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        student_name=student_name,
        chapter_title="Chapter",
        questions=questions,
        user_answers=verified_answers,
        total_time=len(questions) * 30
    )
    
    ReportGenerator.print_report_summary(report)
    return report


# =====================================================
# EXAMPLE 3: Custom Answer Analysis & Detailed Feedback
# =====================================================

def analyze_specific_wrong_answers(
    question: Question,
    wrong_answer: str
) -> dict:
    """
    Analyze why a specific answer is wrong and generate feedback.
    
    Example:
        question = questions[0]
        analysis = analyze_specific_wrong_answers(question, "wrong_option")
    """
    logger.info(f"Analyzing wrong answer for Q{question.id}")
    
    exp_gen = ExplanationGenerator(client)
    
    # Generate feedback
    feedback = exp_gen.generate_wrong_answer_feedback(
        question_text=question.question_text,
        user_answer=wrong_answer,
        correct_answer=question.correct_answer,
        original_explanation=question.explanation
    )
    
    analysis = {
        "question_id": question.id,
        "question_text": question.question_text,
        "user_answer": wrong_answer,
        "correct_answer": question.correct_answer,
        "difficulty": question.difficulty.value,
        "keywords": question.keywords,
        "original_explanation": question.explanation,
        "ai_feedback": feedback,
        "analyzed_at": datetime.now().isoformat()
    }
    
    return analysis


# =====================================================
# EXAMPLE 4: Performance Analytics & Insights
# =====================================================

def generate_performance_insights(reports: List[dict]) -> dict:
    """
    Analyze multiple test reports to generate insights.
    
    Example:
        insights = generate_performance_insights([report1, report2, report3])
    """
    logger.info(f"Generating insights from {len(reports)} reports")
    
    total_tests = len(reports)
    avg_score = sum(r['score_percentage'] for r in reports) / total_tests
    highest_score = max(r['score_percentage'] for r in reports)
    lowest_score = min(r['score_percentage'] for r in reports)
    
    # Aggregate difficulty performance
    difficulty_performance = {"easy": [], "medium": [], "hard": []}
    for report in reports:
        for difficulty, count in report['difficulty_breakdown'].items():
            difficulty_performance[difficulty].append(count)
    
    # Calculate averages
    difficulty_avg = {
        k: sum(v) / len(v) if v else 0 
        for k, v in difficulty_performance.items()
    }
    
    insights = {
        "total_tests": total_tests,
        "average_score": round(avg_score, 2),
        "highest_score": highest_score,
        "lowest_score": lowest_score,
        "score_range": highest_score - lowest_score,
        "difficulty_performance": difficulty_avg,
        "improvement_trend": "improving" if avg_score > 60 else "needs_improvement",
        "generated_at": datetime.now().isoformat()
    }
    
    return insights


# =====================================================
# EXAMPLE 5: Generate Questions with Custom Prompt
# =====================================================

def generate_custom_question_set(
    summary: str,
    question_count: int = 10,
    focus_areas: List[str] = None,
    target_difficulty: str = "medium"
) -> List[Question]:
    """
    Generate questions with custom parameters.
    
    Example:
        questions = generate_custom_question_set(
            summary="...",
            question_count=15,
            focus_areas=["definitions", "applications"],
            target_difficulty="hard"
        )
    """
    logger.info(f"Generating {question_count} custom questions (difficulty: {target_difficulty})")
    
    gen = QuestionGenerator(client)
    
    # In a real scenario, you'd customize the prompt
    # For now, use the standard generation
    questions = gen.generate_test_json(summary, num_questions=question_count)
    
    # Filter by difficulty if needed
    if target_difficulty == "easy":
        questions = [q for q in questions if q.difficulty == DifficultyLevel.EASY]
    elif target_difficulty == "hard":
        questions = [q for q in questions if q.difficulty == DifficultyLevel.HARD]
    
    return questions


# =====================================================
# EXAMPLE 6: Compare Student Performance Across Tests
# =====================================================

def compare_student_performance(student_reports: List[dict]) -> dict:
    """
    Compare a student's performance across multiple tests.
    
    Example:
        comparison = compare_student_performance([test1_report, test2_report, test3_report])
    """
    logger.info(f"Comparing performance across {len(student_reports)} tests")
    
    scores = [r['score_percentage'] for r in student_reports]
    times = [r['time_taken'] for r in student_reports]
    
    improvement = scores[-1] - scores[0] if len(scores) > 1 else 0
    
    comparison = {
        "test_count": len(student_reports),
        "scores_progression": scores,
        "average_score": round(sum(scores) / len(scores), 2),
        "improvement": round(improvement, 2),
        "fastest_test": min(times),
        "slowest_test": max(times),
        "average_time": round(sum(times) / len(times), 2),
        "trend": "improving" if improvement > 0 else "declining" if improvement < 0 else "stable",
        "chapters_tested": [r['chapter_title'] for r in student_reports],
        "analysis_date": datetime.now().isoformat()
    }
    
    return comparison


# =====================================================
# EXAMPLE 7: Export Test Data for External Analysis
# =====================================================

def export_comprehensive_test_data(
    questions: List[Question],
    user_answers: List[UserAnswer],
    report_data: dict,
    export_path: str = "comprehensive_export.json"
) -> None:
    """
    Export complete test data including questions, answers, and report.
    
    Example:
        export_comprehensive_test_data(questions, answers, report, "test_export.json")
    """
    logger.info(f"Exporting comprehensive test data to {export_path}")
    
    export_data = {
        "metadata": {
            "export_date": datetime.now().isoformat(),
            "export_format": "1.0",
            "question_count": len(questions),
            "answer_count": len(user_answers)
        },
        "questions": [q.to_dict() for q in questions],
        "user_responses": [
            {
                "question_id": ua.question_id,
                "selected_answer": ua.selected_answer,
                "is_correct": ua.is_correct,
                "time_taken": ua.time_taken,
                "explanation": ua.explanation
            }
            for ua in user_answers
        ],
        "report": report_data
    }
    
    with open(export_path, 'w') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Export complete: {export_path}")


# =====================================================
# EXAMPLE 8: Validation & Quality Assurance
# =====================================================

def validate_test_quality(questions: List[Question]) -> dict:
    """
    Validate test questions for quality and completeness.
    
    Example:
        quality_report = validate_test_quality(questions)
    """
    logger.info(f"Validating quality of {len(questions)} questions")
    
    issues = []
    
    for q in questions:
        # Check question text
        if not q.question_text or len(q.question_text) < 10:
            issues.append(f"Q{q.id}: Question text too short")
        
        # Check options
        if not q.options or len(q.options) < 2:
            issues.append(f"Q{q.id}: Insufficient options")
        
        # Check correct answer exists in options
        if q.correct_answer not in q.options:
            issues.append(f"Q{q.id}: Correct answer not in options")
        
        # Check explanation
        if not q.explanation or len(q.explanation) < 20:
            issues.append(f"Q{q.id}: Explanation too short")
        
        # Check keywords
        if not q.keywords or len(q.keywords) < 2:
            issues.append(f"Q{q.id}: Insufficient keywords")
    
    quality_report = {
        "total_questions": len(questions),
        "quality_issues": issues,
        "issue_count": len(issues),
        "quality_score": round((1 - len(issues) / len(questions)) * 100, 2) if questions else 0,
        "validation_date": datetime.now().isoformat()
    }
    
    return quality_report


# =====================================================
# MAIN DEMO
# =====================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 ADVANCED USAGE EXAMPLES")
    print("=" * 70)
    
    # Example 1: Batch Generation
    print("\n1️⃣  Batch Generate Questions")
    print("-" * 70)
    sample_chapters = [
        {
            "id": 1,
            "title": "Photosynthesis",
            "summary": "Photosynthesis is the process by which plants convert light energy into chemical energy..."
        },
        {
            "id": 2,
            "title": "Respiration",
            "summary": "Cellular respiration is the process by which cells break down glucose to produce ATP..."
        }
    ]
    # Uncomment to run: batch_generate_questions(sample_chapters)
    
    # Example 2: Performance Insights
    print("\n2️⃣  Generate Performance Insights")
    print("-" * 70)
    sample_reports = [
        {"score_percentage": 75, "difficulty_breakdown": {"easy": 3, "medium": 3, "hard": 1}},
        {"score_percentage": 82, "difficulty_breakdown": {"easy": 3, "medium": 4, "hard": 1}},
        {"score_percentage": 88, "difficulty_breakdown": {"easy": 4, "medium": 4, "hard": 2}}
    ]
    insights = generate_performance_insights(sample_reports)
    print(json.dumps(insights, indent=2))
    
    # Example 3: Student Comparison
    print("\n3️⃣  Compare Student Performance")
    print("-" * 70)
    sample_student_reports = [
        {
            "score_percentage": 65,
            "time_taken": 900,
            "chapter_title": "Chapter 1"
        },
        {
            "score_percentage": 75,
            "time_taken": 800,
            "chapter_title": "Chapter 2"
        },
        {
            "score_percentage": 85,
            "time_taken": 700,
            "chapter_title": "Chapter 3"
        }
    ]
    comparison = compare_student_performance(sample_student_reports)
    print(json.dumps(comparison, indent=2))
    
    print("\n" + "=" * 70)
    print("✅ Examples completed!")
    print("=" * 70)
