import os
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from dotenv import load_dotenv
import json
import uvicorn

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Quiz Generator API",
    description="An API that generates quiz questions based on topic, difficulty, and other parameters",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify allowed origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer: str
    explanation: str

class QuizRequest(BaseModel):
    topic: str = Field(..., description="Subject or topic for the quiz questions")
    difficulty: str = Field(..., description="Difficulty level: easy, medium, or hard")
    time_per_question: int = Field(..., description="Time allowed for each question in seconds")
    num_questions: int = Field(..., description="Number of questions to generate", ge=1, le=20)

class QuizResponse(BaseModel):
    questions: List[QuizQuestion]
    topic: str
    difficulty: str
    time_per_question: int

# Get API key from environment variables
def get_groq_api_key():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ API key not found")
    return api_key

# Initialize Groq client with model
def get_groq_client(api_key: str = Depends(get_groq_api_key)):
    return ChatGroq(
        api_key=api_key,
        model_name="llama3-70b-8192"  # You can change this to your preferred model
    )

@app.get("/")
async def root():
    return {"message": "Welcome to the Quiz Generator API"}

@app.post("/generate-quiz", response_model=QuizResponse)
async def generate_quiz(
    request: QuizRequest,
    groq_client: ChatGroq = Depends(get_groq_client)
):
    try:
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional quiz creator specialized in generating high-quality quiz questions.
                       Create well-formatted quiz questions based on the provided parameters.
                       Each question must have 4 options with exactly one correct answer.
                       Include a brief explanation for the correct answer."""),
            ("human", """Please create {num_questions} {difficulty} level quiz questions about {topic}.
                      Each question should be answerable within {time_per_question} seconds.
                      
                      Return the questions in JSON format with the following structure:
                      [
                        {{
                          "question": "Question text",
                          "options": ["Option A", "Option B", "Option C", "Option D"],
                          "correct_answer": "The correct option text",
                          "explanation": "Brief explanation for the correct answer"
                        }},
                        ...
                      ]
                      
                      Do not include any text before or after the JSON.""")
        ])
        
        # Create chain and run
        chain = prompt | groq_client | StrOutputParser()
        
        result = chain.invoke({
            "topic": request.topic,
            "difficulty": request.difficulty,
            "time_per_question": request.time_per_question,
            "num_questions": request.num_questions
        })
        
        # Parse the result to ensure valid JSON
        try:
            questions_data = json.loads(result)
            
            # Validate the structure of each question
            questions = []
            for q in questions_data:
                question = QuizQuestion(
                    question=q["question"],
                    options=q["options"],
                    correct_answer=q["correct_answer"],
                    explanation=q["explanation"]
                )
                questions.append(question)
            
            return QuizResponse(
                questions=questions,
                topic=request.topic,
                difficulty=request.difficulty,
                time_per_question=request.time_per_question
            )
                
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Failed to parse response from LLM")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating quiz: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# For local development
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)