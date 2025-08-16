from dotenv import load_dotenv
import os 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json
import re
from io import StringIO
import sys

# Load environment variables
load_dotenv()
API_KEY = os.environ.get('OPENAI_API_KEY')

# Initialize OpenAI client
if API_KEY:
    client = OpenAI(api_key=API_KEY)

# Page configuration
st.set_page_config(
    page_title="CSV Data Analyzer with AI",
    page_icon="📊",
    layout="wide"
)

def get_dataframe_info(df):
    """Get comprehensive information about the dataframe"""
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "sample_data": df.head(3).to_dict('records'),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
        "memory_usage": df.memory_usage(deep=True).sum()
    }
    
    # Add basic statistics for numeric columns
    if info["numeric_columns"]:
        info["numeric_stats"] = df[info["numeric_columns"]].describe().to_dict()
    
    return info

def generate_code_with_openai(query, df_info):
    """Generate pandas/visualization code using OpenAI"""
    
    prompt = f"""
You are a data analysis expert. Generate Python code to answer the user's query about their CSV data.

Dataset Information:
- Shape: {df_info['shape']} (rows, columns)
- Columns: {df_info['columns']}
- Data types: {df_info['dtypes']}
- Numeric columns: {df_info['numeric_columns']}
- Categorical columns: {df_info['categorical_columns']}
- Sample data: {df_info['sample_data']}

User Query: {query}

Generate Python code that:
1. Uses pandas operations on a dataframe called 'df'
2. Creates appropriate visualizations using matplotlib, seaborn, or plotly
3. Provides insights and analysis
4. Returns results that can be displayed in Streamlit

Requirements:
- Use only pandas, numpy, matplotlib, seaborn, plotly for analysis and visualization
- Make sure the code is safe to execute
- Include comments explaining the analysis
- If creating plots, use st.pyplot() for matplotlib/seaborn or st.plotly_chart() for plotly
- Structure your response to show both code and explanations

Return your response in this JSON format:
{{
    "code": "# Python code here",
    "explanation": "Explanation of what the code does and insights",
    "visualization_type": "type of visualization created (if any)"
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a data analysis expert who generates clean, safe Python code for data analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        response_text = response.choices[0].message.content
        
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            # Fallback: treat entire response as code
            return {
                "code": response_text,
                "explanation": "Generated analysis code",
                "visualization_type": "custom"
            }
    
    except Exception as e:
        st.error(f"Error generating code: {str(e)}")
        return None

def safe_execute_code(code, df):
    """Safely execute the generated code"""
    try:
        # Create a safe execution environment
        safe_globals = {
            'df': df,
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'px': px,
            'go': go,
            'st': st,
            '__builtins__': {}
        }
        
        # Capture stdout for any print statements
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        # Execute the code
        exec(code, safe_globals)
        
        # Restore stdout
        sys.stdout = old_stdout
        output = captured_output.getvalue()
        
        return True, output
        
    except Exception as e:
        sys.stdout = old_stdout
        return False, str(e)

def main():
    st.title("📊 CSV Data Analyzer with AI")
    st.markdown("Upload your CSV file and ask questions to get insights and visualizations!")
    
    # Check for API key
    if not API_KEY:
        st.error("⚠️ OpenAI API key not found! Please set your OPENAI_API_KEY in the environment variables.")
        st.info("Create a `.env` file with: `OPENAI_API_KEY=your_api_key_here`")
        return
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Upload CSV File")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file to analyze"
        )
        
        if uploaded_file is not None:
            try:
                # Load the CSV file
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully!")
                st.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Store in session state
                st.session_state.df = df
                st.session_state.df_info = get_dataframe_info(df)
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    # Main content area
    if 'df' in st.session_state:
        df = st.session_state.df
        df_info = st.session_state.df_info
        
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["📋 Data Overview", "🤖 AI Analysis", "📈 Quick Stats"])
        
        with tab1:
            st.header("Data Overview")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Dataset Preview")
                st.dataframe(df.head(10), use_container_width=True)
            
            with col2:
                st.subheader("Dataset Info")
                st.metric("Rows", df_info['shape'][0])
                st.metric("Columns", df_info['shape'][1])
                st.metric("Memory Usage", f"{df_info['memory_usage'] / 1024:.1f} KB")
                
                with st.expander("Column Details"):
                    for col, dtype in df_info['dtypes'].items():
                        null_count = df_info['null_counts'][col]
                        st.text(f"{col}: {dtype} ({null_count} nulls)")
        
        with tab2:
            st.header("🤖 AI-Powered Analysis")
            
            # Query input
            query = st.text_area(
                "Ask a question about your data:",
                placeholder="e.g., 'Show me the correlation between numerical columns' or 'Create a histogram of the age column'",
                height=100
            )
            
            if st.button("🔍 Analyze", type="primary"):
                if query:
                    with st.spinner("🤖 AI is analyzing your query..."):
                        # Generate code using OpenAI
                        result = generate_code_with_openai(query, df_info)
                        
                        if result:
                            st.subheader("📝 Analysis Explanation")
                            st.write(result['explanation'])
                            
                            st.subheader("💻 Generated Code")
                            with st.expander("View Generated Code", expanded=False):
                                st.code(result['code'], language='python')
                            
                            st.subheader("📊 Results")
                            # Execute the generated code
                            success, output = safe_execute_code(result['code'], df)
                            
                            if success:
                                if output:
                                    st.text("Output:")
                                    st.text(output)
                            else:
                                st.error(f"Error executing code: {output}")
                else:
                    st.warning("Please enter a query to analyze.")
        
        with tab3:
            st.header("📈 Quick Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Numeric Columns")
                if df_info['numeric_columns']:
                    numeric_df = df[df_info['numeric_columns']]
                    st.dataframe(numeric_df.describe(), use_container_width=True)
                else:
                    st.info("No numeric columns found")
            
            with col2:
                st.subheader("Missing Values")
                missing_data = df.isnull().sum()
                missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
                
                if not missing_data.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    missing_data.plot(kind='bar', ax=ax)
                    ax.set_title('Missing Values by Column')
                    ax.set_ylabel('Count')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                else:
                    st.success("No missing values found!")
            
            # Quick visualizations
            if df_info['numeric_columns']:
                st.subheader("Quick Visualizations")
                
                selected_cols = st.multiselect(
                    "Select columns for correlation heatmap:",
                    df_info['numeric_columns'],
                    default=df_info['numeric_columns'][:5] if len(df_info['numeric_columns']) >= 5 else df_info['numeric_columns']
                )
                
                if len(selected_cols) >= 2:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    correlation_matrix = df[selected_cols].corr()
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                    ax.set_title('Correlation Heatmap')
                    st.pyplot(fig)
    
    else:
        # Welcome message when no file is uploaded
        st.info("👈 Please upload a CSV file from the sidebar to get started!")
        
        st.markdown("""
        ### 🚀 What you can do with this app:
        
        - **Upload CSV files** and get instant overview of your data
        - **Ask natural language questions** like:
          - "Show me the distribution of values in column X"
          - "Create a scatter plot between column A and B"
          - "What are the top 10 categories in column Y?"
          - "Find correlations between numeric columns"
        - **Get AI-generated insights** and visualizations
        - **View quick statistics** and data quality metrics
        
        ### 💡 Example queries to try:
        - "Create a histogram of the price column"
        - "Show me a box plot for each category"
        - "What's the correlation between age and income?"
        - "Group the data by category and show average values"
        """)

if __name__ == "__main__":
    main()
