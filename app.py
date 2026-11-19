
import streamlit as st
import pandas as pd
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

st.set_page_config(page_title="AI Analytics Dashboard", layout="wide")

st.title("AI-Assisted Learning Analytics Platform")
st.caption("Learning behavior analytics with LLM-based interpretation")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    with st.expander("View Raw Data (First 100 Rows)"):
        st.dataframe(df.head(100))

    total_records = len(df)

    completed = df[df["status"].isin(["submitted", "completed"])].shape[0]
    completion_rate = completed / total_records if total_records > 0 else 0

    activity = df.groupby("user_id").size()
    engagement_score = activity.mean()

    if "duration" in df.columns:
        avg_duration = df["duration"].mean()
    else:
        avg_duration = 0

    st.subheader("Analytics Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Completion Rate", f"{completion_rate:.2f}")

    with col2:
        st.metric("Number of Users", len(activity))

    with col3:
        st.metric("Engagement Score", f"{engagement_score:.2f}")

    with col4:
        st.metric("Avg Duration (s)", f"{avg_duration:.1f}")

    st.subheader("Activity Distribution")
    st.bar_chart(df["activity_type"].value_counts())

    st.subheader("User Activity")
    st.bar_chart(activity)

    st.subheader("Engagement Level Distribution")

    user_engagement = activity.reset_index()
    user_engagement.columns = ["user_id", "activity_count"]

    def classify_engagement(count):
        if count < 15:
            return "Low"
        elif count < 25:
            return "Moderate"
        else:
            return "High"

    user_engagement["engagement_level"] = (
        user_engagement["activity_count"].apply(classify_engagement)
    )

    level_counts = (
        user_engagement["engagement_level"]
        .value_counts()
        .reindex(["Low", "Moderate", "High"], fill_value=0)
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Low Engagement", level_counts["Low"])

    with col2:
        st.metric("Moderate Engagement", level_counts["Moderate"])

    with col3:
        st.metric("High Engagement", level_counts["High"])

    engagement_chart = pd.DataFrame({
        "Number of Users": [
            level_counts["Low"],
            level_counts["Moderate"],
            level_counts["High"]
        ]
    }, index=["Low", "Moderate", "High"])

    st.bar_chart(engagement_chart)

    with st.expander("View User Engagement Details"):
        st.dataframe(
            user_engagement.sort_values(
                "activity_count", ascending=False
            ),
            use_container_width=True
        )

    st.subheader("AI Interpretation")

    def generate_llm_feedback(rate, engagement, duration, users):
        prompt = f"""
        You are an educational analytics assistant.

        Completion rate: {rate}
        Engagement score: {engagement}
        Average duration: {duration}
        Number of users: {users}

        Provide a short professional summary (2-3 sentences).
        """

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    if st.button("Generate AI Feedback"):
        with st.spinner("Generating AI interpretation..."):
            try:
                feedback = generate_llm_feedback(
                    completion_rate,
                    engagement_score,
                    avg_duration,
                    len(activity)
                )
                st.success(feedback)

            except Exception as e:
                st.error(f"LLM error: {e}")

                if completion_rate < 0.5:
                    st.warning("Low engagement detected.")
                elif completion_rate < 0.8:
                    st.warning("Moderate engagement observed.")
                else:
                    st.warning("High engagement detected.")

else:
    st.info("Upload a CSV file to start.")
