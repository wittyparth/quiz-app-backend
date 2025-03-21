# Quiz Generator API

A production-ready REST API that dynamically generates customized quiz questions on any topic using LLM technology.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-teal.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1.4-orange.svg)

## 📋 Overview

This API service leverages the Groq LLM platform through LangChain to create tailored quiz questions based on user-specified parameters. Perfect for educational applications, trivia games, or learning platforms.

### Key Features

- **Customizable Questions**: Generate questions on any topic or subject
- **Difficulty Levels**: Choose from easy, medium, or hard difficulties
- **Time-Aware**: Set specific time limits per question
- **Flexible Quantity**: Control exactly how many questions to generate (1-20)
- **Comprehensive Responses**: Each question includes multiple options, correct answer, and explanation

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Groq API key ([Sign up here](https://console.groq.com/))

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/quiz-generator-api.git
   cd quiz-generator-api
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure your environment:
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

### Running Locally

```bash
python app.py
# or
uvicorn app:app --reload
```

Visit `http://127.0.0.1:8000/docs` for the interactive API documentation.

## 🔧 API Reference

### Generate Quiz Questions

```
POST /generate-quiz
```

#### Request Body

```json
{
  "topic": "Quantum Physics",
  "difficulty": "medium",
  "time_per_question": 45,
  "num_questions": 10
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `topic` | string | Subject matter for questions |
| `difficulty` | string | Question difficulty (easy, medium, hard) |
| `time_per_question` | integer | Time allowed per question (seconds) |
| `num_questions` | integer | Number of questions to generate (1-20) |

#### Response

```json
{
  "questions": [
    {
      "question": "What is the principle of quantum superposition?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Option A",
      "explanation": "Explanation of the correct answer"
    },
    // Additional questions...
  ],
  "topic": "Quantum Physics",
  "difficulty": "medium",
  "time_per_question": 45
}
```

## 🐳 Docker Deployment

Build and run the Docker container:

```bash
docker build -t quiz-generator .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key_here quiz-generator
```

## ☁️ Cloud Deployment

### Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/fastapi)

1. Click the button above
2. Set the `GROQ_API_KEY` environment variable
3. Deploy

### AWS, GCP, or Azure

Detailed deployment instructions for cloud platforms are available in the [deployment guide](./docs/deployment.md).

## 🔒 Security Considerations

- The API key is loaded from environment variables for security
- CORS is configured but should be restricted in production
- Rate limiting is recommended for production deployments

## 📊 Performance

- The Llama3-70B model provides an excellent balance of speed and quality
- For higher throughput, consider implementing a caching layer
- Response times average 1-3 seconds per request depending on question count

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For support, email parthasaradhimunakala@gmail.com or open an issue on this repository.