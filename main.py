from dotenv import load_dotenv
import os 
import streamlit as st

load_dotenv()



API_KEY= os.environ['OPENAI_API_KEY']
st.title("Prompt-driven analysis with PandasAI")
