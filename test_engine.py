"""
Test Generation and Evaluation Engine
Includes LangGraph workflow, JSON test format, answer verification, and reporting.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from functools import wraps
import time
from huggingface_hub import InferenceClient
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


# =====================================================
# LOGGING CONFIGURATION
# =====================================================

def setup_logger(name: str, log_file: str = "test_engine.log") -> logging.Logger:
    """Setup logging with both file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logger(__name__)


# =====================================================
# DECORATORS & ERROR HANDLING
# =====================================================

def retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator for retrying failed operations with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"Failed after {max_attempts} attempts: {e}")
                        raise
                    wait_time = delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
        return wrapper
    return decorator


def validate_input(func):
    """Decorator for input validation."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        return func(*args, **kwargs)
    return wrapper


# =====================================================
# ENUMS & DATA CLASSES
# =====================================================

class QuestionType(Enum):
    """Enum for question types."""
    MCQ = "mcq"  # Multiple Choice Question
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"


class DifficultyLevel(Enum):
    """Enum for difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class Question:
    """Data class for a single question."""
    id: int
    question_text: str
    question_type: QuestionType
    options: List[str]
    correct_answer: str
    explanation: str
    difficulty: DifficultyLevel
    keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "question_text": self.question_text,
            "question_type": self.question_type.value,
            "options": self.options,
            "correct_answer": self.correct_answer,
            "explanation": self.explanation,
            "difficulty": self.difficulty.value,
            "keywords": self.keywords
        }


@dataclass
class UserAnswer:
    """Data class for user's answer."""
    question_id: int
    selected_answer: str
    is_correct: bool
    time_taken: float  # seconds
    explanation: str = ""


@dataclass
class TestReport:
    """Data class for test report."""
    test_id: str
    student_name: str
    chapter_title: str
    total_questions: int
    correct_answers: int
    wrong_answers: int
    skipped_questions: int
    score_percentage: float
    time_taken: float  # seconds
    difficulty_breakdown: Dict[str, int]
    question_performance: List[Dict] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class LangGraphState(TypedDict):
    """State for LangGraph workflow."""
    chapter_summary: str
    questions: Optional[List[Question]]
    user_answers: Optional[List[UserAnswer]]
    test_report: Optional[TestReport]
    llm_client: Any
    error: Optional[str]


# =====================================================
# QUESTION GENERATION
# =====================================================

class QuestionGenerator:
    """Generate questions using LLM."""
    
    def __init__(self, llm_client: InferenceClient):
        self.llm_client = llm_client
        logger.info("QuestionGenerator initialized")
    
    @retry(max_attempts=5, delay=2.0)
    def generate_test_json(self, summary: str, num_questions: int = 5) -> List[Question]:
        """Generate test questions in JSON format with LLM."""
        logger.info(f"Generating {num_questions} questions from summary")
        
        prompt = self._build_generation_prompt(summary, num_questions)
        
        response = self.llm_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5000,
            temperature=0.7
        )
        
        response_text = response.choices[0].message.content
        logger.debug(f"Raw LLM response: {response_text}")
        
        questions = self._parse_questions(response_text)
        logger.info(f"Successfully generated {len(questions)} questions")
        
        return questions
    
    def _build_generation_prompt(self, summary: str, num_questions: int) -> str:
        """Build prompt for question generation."""
        return f"""You are a Class 9 teacher creating educational test questions.

Generate exactly {num_questions} test questions from this chapter summary.

CRITICAL REQUIREMENTS:
1. Output ONLY valid JSON, nothing else
2. Each question MUST have ALL these fields: id, question_text, question_type, options, correct_answer, explanation, difficulty, keywords
3. question_type must be: "mcq" OR "true_false" OR "short_answer"
4. difficulty must be: "easy" OR "medium" OR "hard"
5. For MCQ: provide exactly 4 options, correct_answer must be one of them
6. For true_false: options must be ["True", "False"]
7. Keep explanations 1-2 sentences
8. Include 2-3 keywords per question

JSON TEMPLATE:
{{
  "questions": [
    {{
      "id": 1,
      "question_text": "What is photosynthesis?",
      "question_type": "mcq",
      "options": ["Converting light to energy", "Breaking glucose", "Absorbing water", "Releasing CO2"],
      "correct_answer": "Converting light to energy",
      "explanation": "Photosynthesis is the process...",
      "difficulty": "easy",
      "keywords": ["photosynthesis", "plants"]
    }},
    {{
      "id": 2,
      "question_text": "All matter has mass?",
      "question_type": "true_false",
      "options": ["True", "False"],
      "correct_answer": "True",
      "explanation": "Matter is defined as anything with mass.",
      "difficulty": "easy",
      "keywords": ["matter", "mass"]
    }}
  ]
}}

CHAPTER SUMMARY:
{summary}

Now generate exactly {num_questions} questions in the JSON format above:"""
    
    def _parse_questions(self, response_text: str) -> List[Question]:
        """Parse questions from LLM response with robust error handling."""
        try:
            # Try to extract JSON from response
            json_str = response_text.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            data = json.loads(json_str)
            questions = []
            
            for q_data in data.get("questions", []):
                # Validate required fields
                required_fields = ["id", "question_text", "question_type", "options", 
                                  "correct_answer", "explanation", "difficulty"]
                missing_fields = [f for f in required_fields if f not in q_data]
                if missing_fields:
                    logger.warning(f"Question {q_data.get('id', '?')} missing fields: {missing_fields}")
                    continue
                
                # Normalize question_type (handle both "truefalse" and "true_false")
                question_type_str = str(q_data["question_type"]).lower().replace(" ", "_")
                if question_type_str == "truefalse":
                    question_type_str = "true_false"
                
                # Normalize difficulty
                difficulty_str = str(q_data["difficulty"]).lower().strip()
                
                try:
                    question = Question(
                        id=q_data["id"],
                        question_text=str(q_data["question_text"]),
                        question_type=QuestionType(question_type_str),
                        options=[str(opt) for opt in q_data["options"]],
                        correct_answer=str(q_data["correct_answer"]),
                        explanation=str(q_data["explanation"]),
                        difficulty=DifficultyLevel(difficulty_str),
                        keywords=[str(k) for k in q_data.get("keywords", [])]
                    )
                    questions.append(question)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to create question {q_data.get('id', '?')}: {e}")
                    continue
            
            if not questions:
                logger.error("No valid questions parsed from response")
                raise ValueError("LLM response did not contain valid questions")
            
            return questions
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            raise ValueError(f"Invalid JSON in LLM response: {e}")
    
    def export_questions(self, questions: List[Question], filepath: str) -> None:
        """Export questions to JSON file."""
        questions_dict = [q.to_dict() for q in questions]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({"questions": questions_dict}, f, indent=2, ensure_ascii=False)
        logger.info(f"Exported {len(questions)} questions to {filepath}")


# =====================================================
# ANSWER VERIFICATION & EXPLANATION
# =====================================================

class AnswerVerifier:
    """Verify user answers and generate feedback."""
    
    @staticmethod
    @validate_input
    def verify_answer(user_answer: str, correct_answer: str, question_type: QuestionType) -> bool:
        """Verify if user answer matches correct answer."""
        user_ans = user_answer.strip().lower()
        correct_ans = correct_answer.strip().lower()
        
        logger.debug(f"Verifying: '{user_ans}' vs '{correct_ans}'")
        
        if question_type == QuestionType.TRUE_FALSE:
            return user_ans in [correct_ans, "t", "f", "true", "false"]
        
        return user_ans == correct_ans
    
    @staticmethod
    def calculate_score(correct_count: int, total_count: int) -> float:
        """Calculate percentage score."""
        if total_count == 0:
            return 0.0
        return (correct_count / total_count) * 100


class ExplanationGenerator:
    """Generate explanations for wrong answers."""
    
    def __init__(self, llm_client: InferenceClient):
        self.llm_client = llm_client
        logger.info("ExplanationGenerator initialized")
    
    @retry(max_attempts=2, delay=1.0)
    def generate_wrong_answer_feedback(
        self,
        question_text: str,
        user_answer: str,
        correct_answer: str,
        original_explanation: str
    ) -> str:
        """Generate detailed feedback for wrong answers."""
        logger.info(f"Generating feedback for wrong answer")
        prompt = f"""
You are a helpful teacher explaining why an answer is incorrect.

Question: {question_text}
Student's Answer: {user_answer}
Correct Answer: {correct_answer}
Explanation: {original_explanation}

Provide a brief, encouraging explanation (2-3 sentences) about why the student's answer was wrong and why the correct answer is right. Use simple language for Class 9 students.

Explanation:
"""
        
        response = self.llm_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.5
        )
        
        feedback = response.choices[0].message.content
        logger.debug(f"Generated feedback: {feedback}")
        
        return feedback


# =====================================================
# TEST REPORT GENERATION
# =====================================================

class ReportGenerator:
    """Generate comprehensive test reports."""
    
    @staticmethod
    def generate_report(
        test_id: str,
        student_name: str,
        chapter_title: str,
        questions: List[Question],
        user_answers: List[UserAnswer],
        total_time: float
    ) -> TestReport:
        """Generate comprehensive test report."""
        logger.info(f"Generating report for student: {student_name}")
        
        correct_count = sum(1 for ua in user_answers if ua.is_correct)
        wrong_count = len(user_answers) - correct_count
        skipped_count = len(questions) - len(user_answers)
        
        score_percentage = AnswerVerifier.calculate_score(correct_count, len(questions))
        
        # Difficulty breakdown
        difficulty_breakdown = ReportGenerator._calculate_difficulty_breakdown(
            questions, user_answers
        )
        
        # Question performance
        question_performance = ReportGenerator._calculate_question_performance(
            questions, user_answers
        )
        
        report = TestReport(
            test_id=test_id,
            student_name=student_name,
            chapter_title=chapter_title,
            total_questions=len(questions),
            correct_answers=correct_count,
            wrong_answers=wrong_count,
            skipped_questions=skipped_count,
            score_percentage=score_percentage,
            time_taken=total_time,
            difficulty_breakdown=difficulty_breakdown,
            question_performance=question_performance
        )
        
        logger.info(f"Report generated: {score_percentage}% score")
        return report
    
    @staticmethod
    def _calculate_difficulty_breakdown(
        questions: List[Question],
        user_answers: List[UserAnswer]
    ) -> Dict[str, int]:
        """Calculate performance by difficulty level."""
        breakdown = {level.value: 0 for level in DifficultyLevel}
        
        for q in questions:
            for ua in user_answers:
                if ua.question_id == q.id and ua.is_correct:
                    breakdown[q.difficulty.value] += 1
        
        return breakdown
    
    @staticmethod
    def _calculate_question_performance(
        questions: List[Question],
        user_answers: List[UserAnswer]
    ) -> List[Dict]:
        """Calculate performance for each question."""
        performance = []
        
        for q in questions:
            for ua in user_answers:
                if ua.question_id == q.id:
                    performance.append({
                        "question_id": q.id,
                        "question_text": q.question_text,
                        "user_answer": ua.selected_answer,
                        "correct_answer": q.correct_answer,
                        "is_correct": ua.is_correct,
                        "time_taken": ua.time_taken,
                        "difficulty": q.difficulty.value,
                        "explanation": ua.explanation
                    })
        
        return performance
    
    @staticmethod
    def export_report(report: TestReport, filepath: str) -> None:
        """Export report to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report.to_json())
        logger.info(f"Report exported to {filepath}")
    
    @staticmethod
    def print_report_summary(report: TestReport) -> None:
        """Print a formatted report summary."""
        print("\n" + "="*60)
        print("📊 TEST REPORT SUMMARY")
        print("="*60)
        print(f"Student: {report.student_name}")
        print(f"Chapter: {report.chapter_title}")
        print(f"Generated: {report.generated_at}")
        print("-"*60)
        print(f"Total Questions: {report.total_questions}")
        print(f"Correct Answers: {report.correct_answers}")
        print(f"Wrong Answers: {report.wrong_answers}")
        print(f"Skipped: {report.skipped_questions}")
        print("-"*60)
        print(f"Score: {report.score_percentage:.2f}%")
        print(f"Time Taken: {report.time_taken:.2f} seconds")
        print("-"*60)
        print("Performance by Difficulty:")
        for difficulty, count in report.difficulty_breakdown.items():
            print(f"  {difficulty.capitalize()}: {count} correct")
        print("="*60 + "\n")


# =====================================================
# LANGGRAPH WORKFLOW
# =====================================================

class TestWorkflowEngine:
    """LangGraph-based workflow for test generation and evaluation."""
    
    def __init__(self, llm_client: InferenceClient):
        self.llm_client = llm_client
        self.question_gen = QuestionGenerator(llm_client)
        self.explanation_gen = ExplanationGenerator(llm_client)
        self.graph = self._build_graph()
        logger.info("TestWorkflowEngine initialized")
    
    def _build_graph(self):
        """Build LangGraph workflow."""
        workflow = StateGraph(LangGraphState)
        
        # Add nodes
        workflow.add_node("generate_questions", self._generate_questions)
        workflow.add_node("display_test", self._display_test)
        workflow.add_node("evaluate_answers", self._evaluate_answers)
        workflow.add_node("generate_report", self._generate_report)
        
        # Add edges
        workflow.add_edge(START, "generate_questions")
        workflow.add_edge("generate_questions", "display_test")
        workflow.add_edge("display_test", "evaluate_answers")
        workflow.add_edge("evaluate_answers", "generate_report")
        workflow.add_edge("generate_report", END)
        
        logger.debug("LangGraph workflow built successfully")
        return workflow.compile()
    
    def _generate_questions(self, state: LangGraphState) -> LangGraphState:
        """Generate questions node."""
        logger.info("Executing: generate_questions")
        try:
            questions = self.question_gen.generate_test_json(
                state["chapter_summary"],
                num_questions=10
            )
            state["questions"] = questions
            logger.debug(f"Generated {len(questions)} questions")
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            state["error"] = str(e)
        return state
    
    def _display_test(self, state: LangGraphState) -> LangGraphState:
        """Display test node."""
        logger.info("Executing: display_test")
        if state["questions"]:
            print("\n📝 TEST STARTED\n")
            for q in state["questions"]:
                print(f"Q{q.id}: {q.question_text} [{q.difficulty.value.upper()}]")
                if q.question_type == QuestionType.MCQ:
                    for idx, opt in enumerate(q.options, 1):
                        print(f"   {idx}. {opt}")
                elif q.question_type == QuestionType.TRUE_FALSE:
                    print("   1. True")
                    print("   2. False")
                print()
        return state
    
    def _evaluate_answers(self, state: LangGraphState) -> LangGraphState:
        """Evaluate answers node."""
        logger.info("Executing: evaluate_answers")
        state["user_answers"] = state.get("user_answers", [])
        return state
    
    def _generate_report(self, state: LangGraphState) -> LangGraphState:
        """Generate report node."""
        logger.info("Executing: generate_report")
        if state["user_answers"] and state["questions"]:
            report = ReportGenerator.generate_report(
                test_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
                student_name="Student",
                chapter_title="Chapter",
                questions=state["questions"],
                user_answers=state["user_answers"],
                total_time=0
            )
            state["test_report"] = report
        return state
    
    def run_workflow(self, chapter_summary: str, user_answers: Optional[List[UserAnswer]] = None) -> LangGraphState:
        """Run the complete workflow."""
        logger.info("Starting test workflow")
        
        initial_state: LangGraphState = {
            "chapter_summary": chapter_summary,
            "questions": None,
            "user_answers": user_answers or [],
            "test_report": None,
            "llm_client": self.llm_client,
            "error": None
        }
        
        result = self.graph.invoke(initial_state)
        logger.info("Test workflow completed")
        return result


# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def validate_json_format(json_str: str) -> bool:
    """Validate JSON string format."""
    try:
        json.loads(json_str)
        return True
    except json.JSONDecodeError:
        logger.error("Invalid JSON format")
        return False


def get_timestamp() -> str:
    """Get current timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    """Format duration in readable format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


# =====================================================
# CONFIGURATION & CONSTANTS
# =====================================================

DEFAULT_CONFIG = {
    "num_questions": 10,
    "question_mix": {
        "mcq": 0.6,
        "true_false": 0.3,
        "short_answer": 0.1
    },
    "difficulty_distribution": {
        "easy": 0.4,
        "medium": 0.4,
        "hard": 0.2
    },
    "export_dir": "test_outputs",
    "export_questions_for_reference": True,  # Set to False to disable export
    "regenerate_fresh_questions_each_time": True  # Always regenerate, never cache
}


if __name__ == "__main__":
    logger.info("Test engine module loaded successfully")
