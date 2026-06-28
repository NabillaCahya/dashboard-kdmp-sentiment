import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("Mencari model yang tersedia untuk API Key ini...\n")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"Bisa dipakai: {m.name}")
except Exception as e:
    print(f"Gagal mengecek model: {e}")