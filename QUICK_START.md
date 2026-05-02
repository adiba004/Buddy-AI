# 🚀 QUICK START GUIDE

## Overview
This is a complete test generation and evaluation system with AI-powered question generation, interactive testing, answer verification, and comprehensive reporting.

---

## 🎯 What's New

### ✨ Key Features Added:

1. **JSON-Based Question Format** - Structured test format with questions, options, answers, and explanations
2. **LangGraph Integration** - Workflow orchestration for test generation and evaluation
3. **Answer Verification** - Verify user answers and provide feedback
4. **Detailed Explanations** - AI-generated feedback for wrong answers
5. **Comprehensive Reports** - Test reports with performance analytics
6. **Industry Standards** - Logging, error handling, retry logic, validation
7. **Advanced Functions** - Batch processing, offline testing, performance analytics

---

## 📦 Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup environment variables (.env file)
SUPABASE_URL=your_url
SUPABASE_SERVICE_ROLE_KEY=your_key
HF_TOKEN=your_token

# 3. Run the test system
python test.py
```

---

## 🎮 Interactive Test Flow

```
┌─────────────────────────────────────────┐
│  1. Enter Student Name                  │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  2. Select Standard/Grade               │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  3. Select Subject                      │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  4. Select Chapter                      │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  5. AI Generates 10 Questions (JSON)   │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  6. Answer Questions Interactively      │
│     - Type option number (MCQ)          │
│     - Type T/F (True/False)             │
│     - Type 'skip' to skip               │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  7. System Verifies Answers             │
│     - Shows if correct/wrong            │
│     - Displays explanation              │
│     - AI feedback for wrong answers     │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  8. Generate Test Report                │
│     - Score percentage                  │
│     - Performance breakdown             │
│     - Time analysis                     │
│     - Recommendations                   │
└─────────────────────────────────────────┘
```

---

## 📝 JSON Question Format

Each question is generated in this format:

```json
{
  "id": 1,
  "question_text": "What is photosynthesis?",
  "question_type": "mcq",
  "options": [
    "Converting light to chemical energy",
    "Breaking down glucose",
    "Absorbing nitrogen",
    "Releasing oxygen only"
  ],
  "correct_answer": "Converting light to chemical energy",
  "explanation": "Photosynthesis is the process by which plants...",
  "difficulty": "medium",
  "keywords": ["photosynthesis", "plants", "light", "energy"]
}
```

---

## 🧪 Example Usage

### Basic Interactive Test
```bash
python test.py
```
Follow the prompts to:
1. Enter your name
2. Select chapter
3. Answer 10 AI-generated questions
4. View results and report

### Using Components Programmatically

```python
from test_engine import QuestionGenerator, AnswerVerifier, ReportGenerator
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()
client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=os.getenv("HF_TOKEN")
)

# Generate questions
gen = QuestionGenerator(client)
questions = gen.generate_test_json("Chapter summary here", num_questions=10)

# Verify answers
is_correct = AnswerVerifier.verify_answer(
    user_answer="Option A",
    correct_answer="Option A",
    question_type=QuestionType.MCQ
)

# Generate report
report = ReportGenerator.generate_report(
    test_id="test_001",
    student_name="John Doe",
    chapter_title="Photosynthesis",
    questions=questions,
    user_answers=verified_answers,
    total_time=600
)

# Export
ReportGenerator.export_report(report, "report.json")
```

---

## 📊 Generated Files

After running a test, you'll have:

```
├── questions_ChapterTitle.json           # Generated questions
├── test_report_StudentName_TIMESTAMP.json # Test results
├── test_engine.log                       # Execution logs
└── (Optional) batch_output/              # Batch processing results
```

---

## 🎓 Test Report Example

```
======================================================================
📊 TEST REPORT SUMMARY
======================================================================
Student: John Doe
Chapter: Photosynthesis
Generated: 2024-01-15T10:30:45.123456

Total Questions: 10
Correct Answers: 8
Wrong Answers: 1
Skipped: 1

Score: 80.00%
Time Taken: 10m 30s

Performance by Difficulty:
  Easy: 3 correct
  Medium: 4 correct
  Hard: 1 correct
======================================================================
```

---

## 🔧 Advanced Features

### 1. Batch Generate Questions for Multiple Chapters
```python
from advanced_examples import batch_generate_questions

chapters = [
    {"id": 1, "title": "Chapter 1", "summary": "..."},
    {"id": 2, "title": "Chapter 2", "summary": "..."}
]
batch_generate_questions(chapters)
```

### 2. Load Pre-generated Questions (Offline Test)
```python
from advanced_examples import run_offline_test_from_file

report = run_offline_test_from_file("questions.json", "Student Name")
```

### 3. Analyze Performance Across Tests
```python
from advanced_examples import compare_student_performance

comparison = compare_student_performance([report1, report2, report3])
```

### 4. Generate Performance Insights
```python
from advanced_examples import generate_performance_insights

insights = generate_performance_insights([report1, report2, report3])
```

### 5. Validate Question Quality
```python
from advanced_examples import validate_test_quality

quality = validate_test_quality(questions)
print(quality["quality_score"])  # Percentage score
```

---

## 🏗️ Architecture

```
test_engine.py (Core Engine)
├── QuestionGenerator          [LLM-based Q generation]
├── AnswerVerifier             [Answer verification logic]
├── ExplanationGenerator       [AI feedback generation]
├── ReportGenerator            [Report creation & export]
├── TestWorkflowEngine         [LangGraph orchestration]
├── Data Classes               [Question, UserAnswer, TestReport]
├── Decorators                 [Retry, validation]
└── Utilities                  [Logging, formatting]

test.py (Main Interface)
├── Database integration       [Supabase]
├── Interactive test flow      [User interaction]
├── Answer collection          [Input handling]
├── Evaluation loop            [Feedback generation]
└── Report generation          [Export & display]

advanced_examples.py (Extended Features)
├── Batch processing
├── Performance analytics
├── Offline testing
├── Quality validation
└── Data export
```

---

## 🎯 Question Types Supported

| Type | Example | Format |
|------|---------|--------|
| **MCQ** | "Which is..." | Select from 4 options |
| **True/False** | "Is water..." | 1 for True, 2 for False |
| **Short Answer** | "Define..." | Free text input |

---

## 📈 Performance Tracking

The system tracks:
- ✅ Score percentage
- ✅ Time per question
- ✅ Total test duration
- ✅ Difficulty-wise performance
- ✅ Individual question performance
- ✅ Improvement trends

---

## 🔐 Data Persistence

All data is:
- 📁 Exported to JSON files
- 📊 Available for analysis
- 🔒 Locally stored
- 📝 Structured and queryable

---

## ⚙️ Configuration

### Default Settings (in `test_engine.py`)
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
    }
}
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: No module 'langgraph'` | Run: `pip install langgraph` |
| `Invalid JSON in LLM response` | The LLM response format changed. Try again. |
| `Connection refused` | Check Supabase credentials in `.env` |
| `Rate limit exceeded` | Wait a moment and retry. Retry logic is built-in. |

---

## 📚 File Structure

```
Child-Friendly AI Mentor/
├── test.py                           [Main interactive interface]
├── test_engine.py                    [Core engine with all components]
├── advanced_examples.py              [Advanced usage patterns]
├── chat.py                           [Existing chat module]
├── create_embeddings.py              [Existing embeddings module]
├── gs.py                             [Existing utilities]
├── requirements.txt                  [Dependencies]
├── TEST_SYSTEM_DOCUMENTATION.md      [Full documentation]
├── QUICK_START.md                    [This file]
├── .env                              [Environment variables]
├── test_engine.log                   [Execution logs]
├── questions_*.json                  [Generated questions]
└── test_report_*.json                [Test reports]
```

---

## 🚀 Getting Started (5 Minutes)

### Step 1: Install
```bash
pip install -r requirements.txt
```

### Step 2: Configure
Create `.env` file:
```
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_key
HF_TOKEN=your_huggingface_token
```

### Step 3: Run
```bash
python test.py
```

### Step 4: Follow Prompts
- Enter name
- Select chapter
- Answer questions
- View report

**That's it! 🎉**

---

## 💡 Key Features Highlights

✅ **AI-Powered Generation** - LLM generates contextual questions
✅ **JSON Format** - Structured, exportable question format
✅ **Interactive Interface** - User-friendly test taking
✅ **Smart Verification** - Answer checking with explanations
✅ **Detailed Feedback** - AI-generated explanations for errors
✅ **Comprehensive Reports** - Performance analytics and insights
✅ **LangGraph Integration** - Professional workflow orchestration
✅ **Error Handling** - Retry logic and validation
✅ **Logging** - Complete execution tracking
✅ **Batch Processing** - Handle multiple chapters
✅ **Data Export** - JSON-based persistence
✅ **Performance Analytics** - Track improvement over time

---

## 🎓 Next Steps

1. **Run your first test** → `python test.py`
2. **Explore advanced features** → See `advanced_examples.py`
3. **Customize** → Edit prompts in `test_engine.py`
4. **Integrate** → Use components in your own code
5. **Scale** → Batch process multiple chapters

---

## 📞 Support

Refer to:
- **Full Documentation** → `TEST_SYSTEM_DOCUMENTATION.md`
- **Code Examples** → `advanced_examples.py`
- **Logs** → `test_engine.log`

---

**Happy Testing! 🎯**
