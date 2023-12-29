import pandas as pd
import os
import streamlit as st
import openai
from dotenv import load_dotenv
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# Load environment variables
env_path = Path('.env') # Change with your .env file
load_dotenv(dotenv_path=env_path,override=True)

openai.api_type = "azure"
openai.api_key = os.getenv('OPENAI_API_KEY') 
openai.api_base = os.getenv('OPENAI_API_BASE') 
openai.api_version = "2023-03-15-preview"
COMPLETIONS_MODEL = os.environ["COMPLETIONS_MODEL"]



# Enhanced function to process a claim using GPT
def process_claim_with_gpt(claim_details):
    prompt = (
        "You are an AI assistant specialized in insurance fraud detection. Using the claim details provided, evaluate the claim for potential fraud."
        "Please follow these steps:"

        "1. Identify any inconsistencies or unusual patterns in the claim details."
        "2. Compare these findings against common indicators of fraudulent insurance claims."
        "3. Rate the likelihood of fraud on a scale from 'Very Unlikely' to 'Very Likely'."
        "4. Conclude with an overall evaluation, categorizing the claim as 'Normal', 'Suspicious', or 'Highly Suspicious', and explain your reasoning."

        "Claim Details: \n\n" + claim_details 
    )
    response = openai.Completion.create(
        engine=COMPLETIONS_MODEL,
        prompt=prompt,
        temperature=0.5,
        max_tokens=500
    )
    return response.choices[0].text.strip()

def main():

    @st.cache_data
    def load_data():
        # Get the current script's directory
        script_dir = os.path.dirname(__file__)
        # Join the script's directory with the relative file path
        rel_path = "data/insurance_claims.csv"
        abs_file_path = os.path.join(script_dir, rel_path)
        data = pd.read_csv(abs_file_path)
        return data
    df = load_data()

    # Streamlit app interface
    st.title('🤖 AI-Powered Claims Processing App 🔎')
    st.write('Select a claim to process and analyse.👇')

    

    # Dynamic Sidebar Filters
    incident_type = st.sidebar.selectbox("Select Incident Type", options=["All"] + list(df['incident_type'].unique()))
    if incident_type != "All":
        df = df[df['incident_type'] == incident_type]

    # Display a subset of claims data
    st.dataframe(df)  # Interactive table

    # Display an interactive table and allow selection
    st.subheader('📊 Claims Data')
    selected_indices = st.multiselect('Select claims to process (by index):', df.index, default=None)
    selected_rows = df.loc[selected_indices]

    # Displaying the selected data
    st.write('Selected Claims:')
    st.dataframe(selected_rows)

    # Process selected claims
    if st.button('🚀 Process Selected Claims'):
        for index, selected_claim in selected_rows.iterrows():
            # Constructing a detailed claim description using all columns
            claim_details = ', '.join([f"{col}: {selected_claim[col]}" for col in df.columns])
            processed_claim = process_claim_with_gpt(claim_details)
            st.subheader(f'GPT Model Analysis and Fraud Assessment for Claim {index}:')
            st.write(processed_claim)

    # Footer with emojis
    st.write("---")
    st.write("💡 Powered by Streamlit and OpenAI GPT 🤖")

    st.sidebar.markdown("### About")
    st.sidebar.info("This app uses Azure OpenAI's GPT model and Streamlit for processing insurance claims. "
                "It's an innovative approach to analyse and detect potential fraud in claims data.")

if __name__ == "__main__":
    main()