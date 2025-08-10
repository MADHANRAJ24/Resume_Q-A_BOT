import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import PyPDF2
import re

# Load API key from .env
load_dotenv(override=True)

# Initialize Groq LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

st.set_page_config(page_title="Resume Q&A BOT")
st.title("Resume Based Interview Q&A Bot")

# File uploader
upload_file = st.file_uploader("Upload your resume", type=["pdf"])

def extract_skills(text):
    """A simple skill extractor (can be improved with NLP)"""
    skills_list = ["python", "java", "sql", "machine learning", "deep learning", "data analysis",
                   "excel", "pandas", "numpy", "tensorflow", "pytorch", "communication", "leadership"]
    found = set()
    for skill in skills_list:
        if re.search(rf"\b{skill}\b", text, re.IGNORECASE):
            found.add(skill.title())
    return ", ".join(sorted(found)) if found else "Not specified"

if upload_file is not None:
    pdf_reader = PyPDF2.PdfReader(upload_file)
    resume_text = ""

    for page in pdf_reader.pages:
        resume_text += page.extract_text()

    skills = extract_skills(resume_text)

    num_qus = st.slider("Number of questions", 1, 25, 7)
    difficulty = st.selectbox("Question difficulty level", ["Beginner", "Intermediate", "Advanced", "Mixed"])
    question_type = st.multiselect("Select question type(s)", ["Technical", "Behavioral", "General"], default=["Technical"])

    if st.button("Generate Q&A"):
        template = """
        You are an expert technical interviewer. Based on the following resume, generate {num_qus} {difficulty} level 
        interview questions with answers.

        Resume Skills Detected: {skills}
        Selected Question Types: {question_type}

        Rules:
        - Balance question types evenly unless the user specifies fewer types
        - Focus questions on the skills and experiences mentioned in the resume
        - Provide answers that demonstrate depth and relevance
        - Explain why each question is relevant
        - Provide one practical tip for answering

        Output in Markdown table with the following columns:
        | Question Type | Question | Suggested Answer | Why Asked | Pro Tips |

        Resume Content:
        {resume_text}
        """

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm

        response = chain.invoke({
            "num_qus": num_qus,
            "difficulty": difficulty,
            "skills": skills,
            "question_type": ", ".join(question_type),
            "resume_text": resume_text
        })

        st.subheader("Generated Q&A")
        st.markdown(response.content)
