# ==============================
# PERSONAL FINANCE AI - IBM Granite (with PDF & Image Uploads)
# ==============================

# ===== CELL 1: Install & Import =====
!pip install -q transformers accelerate gradio bitsandbytes plotly pdfplumber pytesseract Pillow

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import re
import time
import gc
import warnings
import pdfplumber
import pytesseract
from PIL import Image
warnings.filterwarnings('ignore')

print("✅ Libraries imported successfully!")
print(f"🔥 PyTorch: {torch.__version__}")
print(f"🚀 CUDA: {torch.cuda.is_available()}")

# ===== CELL 2: Load Hugging Face Model =====
model_id = "ibm-granite/granite-3.2-2b-instruct"

print("⏳ Loading model... this may take 1–2 minutes.")
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_8bit=True,
    torch_dtype=torch.float16
)
print("✅ Model loaded successfully!")

# ===== CELL 3: Helper functions for file extraction =====
def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        text = f"⚠️ Error reading PDF: {e}"
    return text.strip()

def extract_text_from_image(file):
    try:
        img = Image.open(file)
        text = pytesseract.image_to_string(img)
    except Exception as e:
        text = f"⚠️ Error reading Image: {e}"
    return text.strip()

# ===== CELL 4: Finance AI Class =====
class PersonalFinanceAI:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.user_profiles = {}
    
    def create_user_profile(self, name, age, income, expenses, savings_goal, user_type="general"):
        self.user_profiles[name] = {
            "age": age,
            "income": income,
            "expenses": expenses,
            "savings_goal": savings_goal,
            "user_type": user_type,
            "created": datetime.now().isoformat()
        }
        return f"✅ Profile created for {name}!"
    
    def generate_response(self, user, prompt, pdf_file=None, image_file=None, max_tokens=300):
        if not user or user not in self.user_profiles:
            return "❌ Please create your profile first!"
        
        # Extract from PDF/Image
        extracted_text = ""
        if pdf_file:
            extracted_text += extract_text_from_pdf(pdf_file)
        if image_file:
            extracted_text += extract_text_from_image(image_file)

        # Merge extracted content with query
        if extracted_text:
            prompt = f"{prompt}\n\n📄 Content from uploaded file(s):\n{extracted_text}"

        # Enforce education-only disclaimer
        full_prompt = (
            "You are a personal finance educational assistant. "
            "Only answer questions related to education, personal finance, saving, investment, and money management. "
            "If the question is unrelated, reply with: '⚠️ Please ask only educational or finance-related questions.'\n\n"
            f"User: {prompt}\nAssistant:"
        )
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()

# ===== CELL 5: Initialize Finance AI =====
finance_ai = PersonalFinanceAI(model, tokenizer)

# ===== CELL 6: Learning Center & Quiz Generator =====
def get_learning_content(user, topic):
    if not user:
        return "❌ Please create your profile first!"
    
    profile = finance_ai.user_profiles.get(user, {})
    user_type = profile.get("user_type", "general")
    age = profile.get("age", 30)
    income = profile.get("income", 0)
    
    prompt = f"""Create an educational lesson on '{topic}' tailored for a {user_type} ({age} years old).
Include:
1. Key concept explanations
2. Practical examples for income ${income:,.0f}
3. Step-by-step actions
4. Common pitfalls
5. Recommended resources
Always include a disclaimer: "This is for educational purposes only."
"""
    return finance_ai.generate_response(user, prompt, max_tokens=600)

def generate_quiz(user, topic, num_questions=5, difficulty="Medium"):
    if not user:
        return "❌ Please create your profile first!"
    
    prompt = f"""Generate {num_questions} multiple-choice quiz questions on '{topic}'.
- 4 options (A-D)
- Mark the correct answer
- Provide a 1-line explanation
Difficulty: {difficulty}
Always include a disclaimer: "This quiz is for educational purposes only."
"""
    return finance_ai.generate_response(user, prompt, max_tokens=400)

# ===== CELL 7: Gradio Interface =====
def create_finance_interface():
    with gr.Blocks(theme=gr.themes.Soft(), title="Personal Finance AI") as app:
        gr.Markdown("# 🏦 Personal Finance AI Advisor\n### Powered by IBM Granite 3.2 2B\nFor **educational purposes only.**")
        current_user = gr.State("")
        
        with gr.Tabs():
            # Profile Tab
            with gr.TabItem("👤 Profile Setup"):
                name = gr.Textbox(label="Name")
                age = gr.Number(label="Age", value=25)
                income = gr.Number(label="Monthly Income ($)", value=2000)
                expenses = gr.Number(label="Monthly Expenses ($)", value=1500)
                goal = gr.Textbox(label="Savings Goal")
                user_type = gr.Dropdown(["student", "professional", "retired", "general"], value="general", label="User Type")
                create_btn = gr.Button("Create Profile ✅")
                profile_output = gr.Textbox(label="Status")
                create_btn.click(
                    lambda n,a,i,e,g,u: finance_ai.create_user_profile(n,a,i,e,g,u),
                    [name,age,income,expenses,goal,user_type],
                    profile_output
                ).then(lambda n: n, name, current_user)
            
            # Advisor Tab
            with gr.TabItem("💬 Chat Advisor"):
                query = gr.Textbox(label="Your Question", placeholder="Ask about budgeting, saving, or investments...")
                pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
                image_file = gr.File(label="Upload Image", file_types=[".png", ".jpg", ".jpeg"], type="filepath")
                ask_btn = gr.Button("Get Advice")
                response = gr.Textbox(label="Response", lines=10)
                ask_btn.click(lambda u,q,pdf,img: finance_ai.generate_response(u,q,pdf,img), [current_user,query,pdf_file,image_file], response)
            
            # Learning Center
            with gr.TabItem("📚 Learning Center"):
                topic = gr.Dropdown(
                    ["Basics of Personal Finance","Investment Fundamentals","Tax Planning","Budgeting","Retirement Planning"],
                    value="Basics of Personal Finance", label="Choose Topic")
                learn_btn = gr.Button("Get Lesson 📖")
                lesson_out = gr.Textbox(lines=12, label="Lesson")
                learn_btn.click(get_learning_content, [current_user, topic], lesson_out)
                
                gr.Markdown("### 📝 Quiz Generator")
                quiz_topic = gr.Textbox(label="Quiz Topic", placeholder="e.g., Investment Fundamentals")
                num_q = gr.Slider(1,10,5,label="Number of Questions")
                diff = gr.Dropdown(["Easy","Medium","Hard"], value="Medium", label="Difficulty")
                quiz_btn = gr.Button("Generate Quiz")
                quiz_out = gr.Textbox(lines=12, label="Quiz")
                quiz_btn.click(generate_quiz, [current_user, quiz_topic, num_q, diff], quiz_out)
        
        return app

app = create_finance_interface()
app.launch(share=True)
