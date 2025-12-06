import streamlit as st
import pandas as pd
import plotly.express as px
from google import genai
from PIL import Image
import json
import datetime
import io

# --- 0. CONFIGURATION & SETUP ---

# <--- PLACE THE GEMINI CLIENT CONFIGURATION HERE (Global Initialization) --->
try:
    # 1. Access the Key: Assumes your secrets file (.streamlit/secrets.toml) has:
    #    [gemini]
    #    api_key = "YOUR_API_KEY"
    API_KEY = st.secrets["gemini"]["api_key"]
    
    # 2. Configure the Client: Initialize the client once globally.
    client = genai.Client(api_key=API_KEY)
    
except KeyError:
    st.error("🚨 Configuration Error: Gemini API key not found. Ensure it's set in `.streamlit/secrets.toml` under the `[gemini]` section.")
    st.stop()
except Exception as e:
    st.error(f"Error configuring Gemini client: {e}")
    st.stop()
# <--- END OF CLIENT CONFIGURATION --->


# Function to analyze the image (Module 1 - requires the initialized client)
def analyze_road_hazard(uploaded_file, client):
    """Uses Gemini to analyze an image for road hazards and extract hazard/severity."""
    img = Image.open(uploaded_file)
    prompt = """
    Analyze this image for any road infrastructure defects (Potholes/Cracks, Broken streetlights, Waterlogging/Drainage issues, or Traffic sign obstruction). 
    
    Respond STRICTLY in the following JSON format only, providing a single hazard type that best describes the issue and an appropriate severity score (Low, Medium, or High).

    {
        "hazard_detected": "[Hazard Type]",
        "severity_score": "[Severity Score]"
    }
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, img],
            config={"response_mime_type": "application/json"}
        )
        analysis = json.loads(response.text)
        return analysis.get("hazard_detected"), analysis.get("severity_score")
    except Exception as e:
        st.error(f"Gemini Analysis Error: {e}")
        return None, None

# Function for Module 2 analysis (FINAL CORRECTED VERSION)
def process_accident_data(df):
    """Processes the DataFrame to generate key insights and charts."""
    
    # Required columns based on your uploaded CSV
    required_cols = ['Local_Authority_(District)', 'Time', 'Junction_Control']
    
    # Check for required columns
    if not all(col in df.columns for col in required_cols):
        st.error(f"Dataset must contain these columns: {', '.join(required_cols)}")
        return None, None, None, None, None 

    # 1. Location Frequency
    location_counts = df['Local_Authority_(District)'].value_counts().reset_index()
    location_counts.columns = ['Location', 'Accident Frequency']
    highest_freq_location = location_counts.iloc[0]['Location']
    fig_location = px.bar(
        location_counts,
        x='Location',
        y='Accident Frequency',
        title='Accident Frequency by Location 📍',
        color='Accident Frequency'
    )

    # 2. Most Common Cause
    cause_counts = df['Junction_Control'].value_counts().reset_index()
    cause_counts.columns = ['Cause', 'Count']
    most_common_cause = cause_counts.iloc[0]['Cause']
    fig_cause = px.pie(
        cause_counts,
        names='Cause',
        values='Count',
        title='Distribution of Accident Causes 💥'
    )

    # 3. Time-of-Day Analysis (Day vs. Night)
    
    # 🚨 FIX for 'nan' values: Create a temporary DataFrame for time analysis, 
    # dropping rows where 'Time' is missing (or null/NaN).
    df_time = df.dropna(subset=['Time']).copy()

    # Initialize fig_time to None in case processing fails
    fig_time = None
    
    try:
        # Process time data on the clean subset
        df_time['Hour'] = df_time['Time'].apply(lambda x: datetime.datetime.strptime(str(x).split(':')[0], '%H').hour)
        
        df_time['Time_of_Day'] = df_time['Hour'].apply(lambda h: 'Day (06:00 - 17:59)' if 6 <= h < 18 else 'Night (18:00 - 05:59)')
        time_of_day_counts = df_time['Time_of_Day'].value_counts().reset_index()
        time_of_day_counts.columns = ['Time_of_Day', 'Count']
        fig_time = px.bar(
            time_of_day_counts,
            x='Time_of_Day',
            y='Count',
            title='Accidents: Day vs. Night ☀️🌙',
            color='Time_of_Day'
        )
    except ValueError:
        st.warning("Skipping Time-of-Day Analysis: Found non-standard time values (not in 'HH:MM' format) after removing missing data.")
        # fig_time remains None
    
    return highest_freq_location, most_common_cause, fig_location, fig_cause, fig_time


# --- STREAMLIT APP LAYOUT ---

st.set_page_config(
    page_title="Road Rakshak 2.0 Dashboard 🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Project Road Rakshak 2.0: Integrated Safety Dashboard 🛣️")
st.markdown("Leveraging Multimodal AI to **see**, **analyze**, and **act** on road safety issues.")
st.markdown("---")

# Initialize session state variables
if 'hazard' not in st.session_state: st.session_state['hazard'] = None
if 'severity' not in st.session_state: st.session_state['severity'] = None
if 'hotspot' not in st.session_state: st.session_state['hotspot'] = None
if 'cause' not in st.session_state: st.session_state['cause'] = None

# --- SIDEBAR FOR MODULE 2 DATA UPLOAD ---
with st.sidebar:
    st.header("Upload Accident Data 🧠")
    st.info("Upload your CSV dataset for Module 2 analysis. Using columns: Local_Authority_(District), Time (HH:MM), Junction_Control.")
    
    uploaded_csv = st.file_uploader(
        "Upload CSV Dataset:",
        type=["csv"],
        key="csv_uploader"
    )

    df_accident = pd.DataFrame()
    if uploaded_csv is not None:
        try:
            df_accident = pd.read_csv(uploaded_csv)
            st.success("CSV dataset loaded successfully.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# --- MAIN DASHBOARD LAYOUT ---

# Create two main columns for a clean look
col1, col2 = st.columns([1, 1])

# --- COLUMN 1: MODULE 1 (VISUAL INTELLIGENCE) ---
with col1:
    st.header("1. Visual Intelligence (The Eyes) 👀")
    st.markdown("Upload a street image for real-time hazard detection.")
    
    uploaded_file = st.file_uploader(
        "Upload a street view image (JPG/PNG):",
        type=["jpg", "png"]
    )

    if uploaded_file is not None:
        try:
            with st.spinner("Analyzing image for infrastructure defects using Gemini..."):
                hazard_detected, severity_score = analyze_road_hazard(uploaded_file, client)
            
            st.image(uploaded_file, caption='Image uploaded successfully.', use_column_width=True)

            if hazard_detected and severity_score:
                st.session_state['hazard'] = hazard_detected
                st.session_state['severity'] = severity_score
                
                st.success(f"✅ AI Analysis Complete (simulated):")
                st.markdown(f"**Hazard Detected:** `{hazard_detected}`")
                
                # Display severity with color-coding
                if severity_score.lower() == 'high':
                    st.error(f"**Estimated Severity Score:** {severity_score} ⚠️")
                elif severity_score.lower() == 'medium':
                    st.warning(f"**Estimated Severity Score:** {severity_score} ❗")
                else:
                    st.success(f"**Estimated Severity Score:** {severity_score} 👍")

            else:
                st.warning("Could not identify a clear hazard or the AI response was invalid. Please check your prompt/model settings.")
                st.session_state['hazard'] = None
                st.session_state['severity'] = None

        except Exception as e:
             st.error(f"An unexpected error occurred during analysis: {e}")


# --- COLUMN 2: MODULE 2 (DATA ANALYTICS) ---
with col2:
    st.header("2. Data Analytics (The Brain) 🧠")
    
    if not df_accident.empty:
        st.subheader("Key Accident Insights")
        
        highest_freq_location, most_common_cause, fig_location, fig_cause, fig_time = process_accident_data(df_accident)

        # Check if the function returned valid data (not the 5 None values)
        if highest_freq_location is not None:
            # Display Key Findings
            st.info(f"**Insight 1 (Hotspot):** The location with the highest accident frequency is **{highest_freq_location}**.")
            st.warning(f"**Insight 2 (Cause):** The most common cause of accidents is **{most_common_cause}**.")

            # Store results for Module 3
            st.session_state['hotspot'] = highest_freq_location
            st.session_state['cause'] = most_common_cause

            # Display Charts in an expandable container
            with st.expander("View Detailed Charts and Raw Data"):
                st.plotly_chart(fig_location, use_container_width=True)
                st.plotly_chart(fig_cause, use_container_width=True)
                
                # Only show time chart if it was successfully generated
                if fig_time is not None:
                     st.plotly_chart(fig_time, use_container_width=True)
                
                st.markdown("---")
                st.dataframe(df_accident)
        else:
             st.warning("Data processing failed. Please ensure the CSV contains 'Local_Authority_(District)', 'Time', and 'Junction_Control' columns.")

    else:
        st.warning("⬅️ Please upload a CSV file in the sidebar to process accident data.")


st.markdown("---")

# --- MODULE 3 (CIVIC REPORTING) ---
st.header("3. Civic Reporting (The Voice) 📢")
st.markdown("Synthesizing data into a formal, actionable complaint letter.")

hazard = st.session_state.get('hazard')
severity = st.session_state.get('severity')
hotspot = st.session_state.get('hotspot')
cause = st.session_state.get('cause')

if hazard and hotspot and severity and cause:
    # 3.1 Task: Generate formal letter
    subject = f"Urgent Repair Required: {hazard} at {hotspot} (Severity: {severity})"

    body = f"""
    Dear Municipal Commissioner,

    I am writing to bring to your urgent attention a significant road safety hazard identified at **{hotspot}**.

    Through a recent visual assessment (Module 1), we have detected:
    **Hazard Type:** {hazard}
    **Estimated Severity:** {severity}

    This issue requires immediate action as historical data (Module 2) indicates this location is a serious **accident hotspot**. Our analysis shows that **{hotspot}** has the highest frequency of accidents in the area, with a common contributing factor being **{cause}**.

    The presence of the reported hazard ({hazard}) directly correlates with and exacerbates the high-risk conditions at this location. We request your immediate intervention to schedule necessary repairs and mitigate the danger to commuters.

    Thank you for your professional, urgent attention to this matter.

    Sincerely,
    Concerned Citizen
    """

    st.success("✅ Formal Complaint Letter Drafted!")
    st.info(f"**Subject Line:** {subject}")

    st.text_area(
        "Formal Complaint Letter (Ready to Email)",
        body,
        height=400
    )

    # 3.2 Output: Downloadable Text File
    st.download_button(
        label="📥 Download Complaint Letter (Text File)",
        data=body,
        file_name="Road_Rakshak_Complaint.txt",
        mime="text/plain"
    )

else:
    st.warning("🚧 Data missing. Please complete Module 1 (upload image) and Module 2 (upload CSV) to generate the full report.")