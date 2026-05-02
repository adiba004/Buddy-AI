# 🎓 Child-Friendly AI Mentor - Test System Documentation

## Overview

This is a comprehensive test generation and evaluation system built with industry-standard practices. It uses LangGraph for workflow orchestration, AI-powered question generation, and detailed performance reporting.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Test Workflow Engine                       │
│                    (LangGraph Based)                         │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Generate       │  │   Display Test  │  │  Evaluate &     │
│   Questions      │  │                 │  │  Generate       │
│   (JSON Format)  │  │  (Interactive)  │  │  Report         │
└──────────────────┘  └─────────────────┘  └─────────────────┘
         │
         └─ LLM Prompt → Structured JSON → Parse & Validate
```

## JSON Test Format

### Question Schema

```json
{
  "questions": [
    {
      "id": 1,
      "question_text": "What is photosynthesis?",
      "question_type": "mcq",
      "options": [
        "Process of converting light to chemical energy",
        "Process of breaking down glucose",
        "Process of absorbing nitrogen",
        "Process of releasing oxygen only"
      ],
      "correct_answer": "Process of converting light to chemical energy",
      "explanation": "Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose...",
      "difficulty": "medium",
      "keywords": ["photosynthesis", "plants", "light", "energy", "chlorophyll"]
    },
    {
      "id": 2,
      "question_text": "Is mitochondria called the powerhouse of the cell?",
      "question_type": "true_false",
      "options": ["True", "False"],
      "correct_answer": "True",
      "explanation": "Yes, mitochondria is known as the powerhouse because it generates ATP through cellular respiration.",
      "difficulty": "easy",
      "keywords": ["mitochondria", "ATP", "cellular respiration"]
    }
  ]
}
```

### Question Types Supported

| Type | Description | Options | Example |
|------|-------------|---------|---------|
| **mcq** | Multiple Choice | 4 options | "Which of the..." |
| **true_false** | True/False | 2 options | "Is water..." |
| **short_answer** | Short Answer | Text input | "Define..." |

### Difficulty Levels

- **easy**: Basic recall and understanding
- **medium**: Application and analysis
- **hard**: Synthesis and evaluation

## Core Components

### 1. **QuestionGenerator**
Generates test questions using LLM.

```python
from test_engine import QuestionGenerator
from huggingface_hub import InferenceClient

client = InferenceClient(model="meta-llama/Llama-3.1-8B-Instruct", token=HF_TOKEN)
gen = QuestionGenerator(client)

questions = gen.generate_test_json(summary, num_questions=10)
gen.export_questions(questions, "questions.json")
```

**Features:**
- ✅ LLM-based generation with retry logic
- ✅ JSON parsing with error handling
- ✅ Export to file
- ✅ Difficulty distribution
- ✅ Keyword extraction

---

### 2. **AnswerVerifier**
Verifies user answers against correct answers.

```python
from test_engine import AnswerVerifier, QuestionType

is_correct = AnswerVerifier.verify_answer(
    user_answer="True",
    correct_answer="True",
    question_type=QuestionType.TRUE_FALSE
)

score = AnswerVerifier.calculate_score(correct_count=8, total_count=10)
print(f"Score: {score}%")  # Output: Score: 80.0%
```

**Features:**
- ✅ Type-aware verification
- ✅ Case-insensitive matching
- ✅ Score calculation
- ✅ Flexible answer matching

---

### 3. **ExplanationGenerator**
Generates detailed feedback for wrong answers.

```python
from test_engine import ExplanationGenerator

exp_gen = ExplanationGenerator(client)

feedback = exp_gen.generate_wrong_answer_feedback(
    question_text="What is photosynthesis?",
    user_answer="Process of breaking down glucose",
    correct_answer="Process of converting light to chemical energy",
    original_explanation="Photosynthesis is..."
)
print(feedback)
```

**Features:**
- ✅ AI-powered feedback generation
- ✅ Student-friendly explanations
- ✅ Retry logic with exponential backoff
- ✅ Encouraging tone

---

### 4. **ReportGenerator**
Generates comprehensive test reports.

```python
from test_engine import ReportGenerator

report = ReportGenerator.generate_report(
    test_id="test_001",
    student_name="John Doe",
    chapter_title="Photosynthesis",
    questions=questions,
    user_answers=verified_answers,
    total_time=600  # seconds
)

ReportGenerator.print_report_summary(report)
ReportGenerator.export_report(report, "report.json")
```

**Report Contains:**
- Total questions and performance
- Correct/wrong/skipped breakdown
- Score percentage
- Time analysis
- Difficulty-wise performance
- Individual question performance
- Detailed explanations

---

### 5. **TestWorkflowEngine (LangGraph)**
Orchestrates the complete test workflow.

```python
from test_engine import TestWorkflowEngine

engine = TestWorkflowEngine(client)
result = engine.run_workflow(chapter_summary)

print(result["test_report"])
```

**Workflow Nodes:**
1. `generate_questions` - AI generates questions
2. `display_test` - Show questions to user
3. `evaluate_answers` - Verify and grade
4. `generate_report` - Create comprehensive report

---

## Utility Functions

### Logging Configuration
```python
from test_engine import setup_logger

logger = setup_logger(__name__, log_file="my_test.log")
logger.info("Test started")
```

### Decorators

**@retry** - Retry failed operations with exponential backoff
```python
@retry(max_attempts=3, delay=1.0)
def risky_operation():
    pass
```

**@validate_input** - Log function calls
```python
@validate_input
def process_data(data):
    pass
```

### Helper Functions
```python
from test_engine import (
    format_duration,      # "5m 30s"
    get_timestamp,        # "2024-01-15 10:30:45"
    validate_json_format  # True/False
)
```

---

## Data Classes

### Question
```python
@dataclass
class Question:
    id: int
    question_text: str
    question_type: QuestionType
    options: List[str]
    correct_answer: str
    explanation: str
    difficulty: DifficultyLevel
    keywords: List[str]
```

### UserAnswer
```python
@dataclass
class UserAnswer:
    question_id: int
    selected_answer: str
    is_correct: bool
    time_taken: float
    explanation: str
```

### TestReport
```python
@dataclass
class TestReport:
    test_id: str
    student_name: str
    chapter_title: str
    total_questions: int
    correct_answers: int
    wrong_answers: int
    skipped_questions: int
    score_percentage: float
    time_taken: float
    difficulty_breakdown: Dict[str, int]
    question_performance: List[Dict]
    generated_at: str
```

---

## Usage Guide

### Basic Test Flow

```python
from test.py import run

# Run interactive test
run()
```

**Step-by-step:**
1. Enter student name
2. Select standard/grade
3. Select subject
4. Select chapter
5. AI generates 10 questions
6. Answer each question
7. View results with explanations
8. Get comprehensive report

---

### Advanced Usage

#### Generate Custom Questions
```python
from test_engine import QuestionGenerator
from huggingface_hub import InferenceClient

client = InferenceClient(model="...", token="...")
gen = QuestionGenerator(client)

summary = "Chapter content..."
questions = gen.generate_test_json(summary, num_questions=15)

# Export
gen.export_questions(questions, "output.json")

# Use with different configurations
for q in questions:
    print(f"Q{q.id}: {q.question_text} ({q.difficulty.value})")
```

#### Batch Process Multiple Chapters
```python
import json
from test_engine import QuestionGenerator, ReportGenerator

client = InferenceClient(...)
gen = QuestionGenerator(client)

chapters = get_all_chapters()

for chapter in chapters:
    summary = chapter.get_summary()
    questions = gen.generate_test_json(summary)
    
    # Save questions
    filename = f"q_{chapter.id}.json"
    gen.export_questions(questions, filename)
```

#### Custom Report Generation
```python
from test_engine import ReportGenerator, TestReport

# Custom report with specific requirements
report = ReportGenerator.generate_report(
    test_id="custom_001",
    student_name="Advanced Student",
    chapter_title="Complex Topic",
    questions=questions,
    user_answers=answers,
    total_time=1200
)

# Print summary
ReportGenerator.print_report_summary(report)

# Export as JSON
ReportGenerator.export_report(report, "detailed_report.json")
```

---

## Configuration

### Default Settings
```python
DEFAULT_CONFIG = {
    "num_questions": 10,
    "question_mix": {
        "mcq": 0.6,           # 60%
        "true_false": 0.3,    # 30%
        "short_answer": 0.1   # 10%
    },
    "difficulty_distribution": {
        "easy": 0.4,          # 40%
        "medium": 0.4,        # 40%
        "hard": 0.2           # 20%
    },
    "export_dir": "test_outputs"
}
```

---

## Industry-Standard Features

### 1. **Error Handling & Logging**
- Comprehensive logging to file and console
- Structured error messages
- Retry logic with exponential backoff

### 2. **Input Validation**
- JSON format validation
- Answer format verification
- Type checking with dataclasses

### 3. **Performance Tracking**
- Time tracking per question
- Total test duration
- Performance metrics by difficulty

### 4. **Data Persistence**
- Export questions to JSON
- Export reports to JSON
- Structured data format

### 5. **Code Quality**
- Type hints throughout
- Docstrings for all functions
- Decorators for cross-cutting concerns
- Clean separation of concerns

### 6. **Scalability**
- LangGraph for workflow orchestration
- Modular component design
- Easy to extend with new question types
- Support for batch processing

---

## Example Output

### Generated Test Questions
```json
{
  "questions": [
    {
      "id": 1,
      "question_text": "What is photosynthesis?",
      "question_type": "mcq",
      "options": [...],
      "correct_answer": "...",
      "explanation": "...",
      "difficulty": "medium",
      "keywords": [...]
    }
  ]
}
```

### Test Report
```json
{
  "test_id": "20240115_103045",
  "student_name": "John Doe",
  "chapter_title": "Photosynthesis",
  "total_questions": 10,
  "correct_answers": 8,
  "wrong_answers": 1,
  "skipped_questions": 1,
  "score_percentage": 80.0,
  "time_taken": 600.0,
  "difficulty_breakdown": {
    "easy": 3,
    "medium": 4,
    "hard": 1
  },
  "question_performance": [...],
  "generated_at": "2024-01-15T10:30:45"
}
```

---

## Files Generated

After running a test, you'll have:

1. **questions_{chapter_title}.json** - Generated questions for reference
2. **test_report_{student_name}_{timestamp}.json** - Detailed test report
3. **test_engine.log** - Complete execution log

---

## Environment Variables

```bash
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
HF_TOKEN=your_huggingface_token
```

---

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the test system
python test.py
```

---

## Troubleshooting

### JSON Parsing Error
**Problem:** "Invalid JSON in LLM response"
**Solution:** The LLM response contains invalid JSON. Check the prompt and try again.

### Supabase Connection Error
**Problem:** "Connection refused"
**Solution:** Verify SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env

### LLM Rate Limit
**Problem:** "Rate limit exceeded"
**Solution:** Retry logic is automatic. Check HF_TOKEN and API limits.

---

## Future Enhancements

- [ ] Multi-language support
- [ ] Adaptive difficulty based on performance
- [ ] Image-based questions
- [ ] Video explanations
- [ ] Leaderboard system
- [ ] Personalized study recommendations
- [ ] Mobile app integration
- [ ] Real-time collaboration

---

## License & Credits

Built with industry best practices for educational AI systems.
Technologies: LangGraph, LLaMA, Supabase, HuggingFace

---

**Last Updated:** January 2024
**Version:** 1.0
