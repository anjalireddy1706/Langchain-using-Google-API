# Integrate our code with LLM
import warnings
import sys
from constants import API_KEY
from langchain_google_genai import GoogleGenerativeAI
import streamlit as st

sys.modules['warnings'] = warnings

llm = GoogleGenerativeAI(
    model="models/gemini-1.5-flash",  # Replace with a valid model from your list_models() output
    google_api_key=API_KEY,
)

st.title("Langchain Demo with Google AI")
input_text = st.text_input("Search the topic you want")

if input_text:
    response = llm(input_text)
    st.write(response)

