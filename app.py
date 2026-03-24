import streamlit as st
import pandas as pd
from openai import OpenAI

# API Configuration
client = OpenAI(
    api_key="YOUR_DEEPSEEK_API_KEY",
    base_url="https://api.deepseek.com"
)

# Page Setup
st.set_page_config(page_title="AI Analytics Dashboard", layout="wide")

st.title("AI-Assisted Learning Analytics Platform")
st.caption("Analytics and LLM-based interpretation")

# File Upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    # Data Loading
    df = pd.read_csv(uploaded_file)

    with st.expander("View Raw Data"):
        st.dataframe(df)

    # Analytics Module
    total_tasks = df[df["activity_type"] == "quiz"].shape[0]
    completed = df[df["status"] == "submitted"].shape[0]

    if total_tasks > 0:
        completion_rate = completed / total_tasks
    else:
        completion_rate = 0

    activity = df.groupby("user_id").size()
    engagement_score = activity.mean()

    # Visualization
    st.subheader("Analytics Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Completion Rate", f"{completion_rate:.2f}")

    with col2:
        st.metric("Number of Users", len(activity))

    with col3:
        st.metric("Engagement Score", f"{engagement_score:.2f}")

    st.bar_chart(activity)

    # AI Module
    st.subheader("AI Interpretation")

    def generate_llm_feedback(rate, engagement, users):
        prompt = f"""
        You are an educational analytics assistant.

        Completion rate: {rate}
        Engagement score: {engagement}
        Number of users: {users}

        Provide a short professional summary (2-3 sentences).
        """

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    # Interaction
    if st.button("Generate AI Feedback"):
        try:
            feedback = generate_llm_feedback(
                completion_rate,
                engagement_score,
                len(activity)
            )
            st.success(feedback)

        except Exception as e:
            st.error(f"LLM error: {e}")

            # Fallback Mechanism
            if completion_rate < 0.5:
                st.warning("Low engagement detected.")
            elif completion_rate < 0.8:
                st.warning("Moderate engagement observed.")
            else:
                st.warning("High engagement detected.")

else:
    st.info("Upload a CSV file to start.")
