import pandas as pd
import os
import streamlit as st
import openai
from dotenv import load_dotenv
from pathlib import Path

env_path = Path('.env') # Change with your .env file
load_dotenv(dotenv_path=env_path,override=True)

openai.api_type = "azure"
openai.api_key = os.getenv('OPENAI_API_KEY') 
openai.api_base = os.getenv('OPENAI_API_BASE') 
openai.api_version = "2023-03-15-preview"
COMPLETIONS_MODEL = os.environ["COMPLETIONS_MODEL"]

@st.cache_data
def load_data():
    # Get the current script's directory
    script_dir = os.path.dirname(__file__)
    # Join the script's directory with the relative file path
    rel_path = "data/generated_insurance_claims_dataset.csv"
    abs_file_path = os.path.join(script_dir, rel_path)
    data = pd.read_csv(abs_file_path)
    return data

df = load_data()

# Enhanced function to process a claim using GPT
def process_claim_with_gpt(claim_text):
    prompt = (
        "You are an AI assistant for processing insurance claims. "
        "Review the following claim and provide a comprehensive analysis. "
        "Check for completeness, coherence, identify any missing or ambiguous information, "
        "assess the claim's validity, and suggest the next steps (approve, deny, escalate): \n\n" +
        claim_text
    )
    response = openai.Completion.create(
        engine=COMPLETIONS_MODEL,
        prompt=prompt,
        temperature=0.5,
        max_tokens=250
    )
    return response.choices[0].text.strip()

# Streamlit app interface
st.title('AI-Powered Claims Processing App')
st.write('Select a claim to process and analyze.')

# Display a subset of claims data
st.write(df[['Claim ID', 'Claimant Name', 'Incident Type', 'Incident Date', 'Claim Amount']])

# User selects a claim to process
selected_claim_id = st.selectbox("Select a claim ID:", df['Claim ID'])
if st.button('Process Claim'):
    selected_claim = df[df['Claim ID'] == selected_claim_id]
    claim_text = f"Claimant: {selected_claim['Claimant Name'].iloc[0]}, " \
                 f"Incident Type: {selected_claim['Incident Type'].iloc[0]}, " \
                 f"Incident Date: {selected_claim['Incident Date'].iloc[0]}, " \
                 f"Claim Amount: {selected_claim['Claim Amount'].iloc[0]}"
    processed_claim = process_claim_with_gpt(claim_text)
    st.subheader('GPT Model Analysis:')
    st.write(processed_claim)