import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
import os
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Flipkart Review Sentiment Classifier",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Flipkart-inspired design
st.markdown("""
<style>
    .main { background-color: #F5F7FA; }
    .stButton>button {
        background-color: #2874F0;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FB641B;
    }
    .stTextArea textarea {
        border: 2px solid #2874F0;
        border-radius: 8px;
        background-color: #000000;
    }
    .stAlert {
        border-radius: 8px;
    }
    .stMetric {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load model from local fine-tuned folder
@st.cache_resource
def load_pipeline():
    try:
        model_dir = "fp-model"
        return pipeline("text-classification", model=model_dir, tokenizer=model_dir)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

clf = load_pipeline()

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Main UI
st.title("🛍️ Flipkart Review Sentiment Classifier")
st.markdown("**Discover the sentiment behind Flipkart reviews!** Enter a review to classify it as **Positive**, **Negative**, or **Neutral**.")

# Input area
review = st.text_area("📝 Write Your Review", height=120, placeholder="e.g., Super fast delivery and amazing product quality!")

# Predict button
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🔍 Analyze Sentiment", type="primary"):
        if not clf:
            st.error("Model not loaded. Please check the model directory.")
        elif review.strip() == "":
            st.warning("Please enter a review to analyze.")
        else:
            with st.spinner("Processing..."):
                try:
                    result = clf(review)[0]
                    label = result['label']
                    confidence = result['score']
                    
                    # Map labels to human-readable sentiment
                    sentiment_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
                    sentiment = sentiment_map.get(label, "Unknown")
                    
                    # Color-coded sentiment display
                    if sentiment == "Positive":
                        st.success(f"**Sentiment:** {sentiment} 😊 | **Confidence:** {confidence:.2f}")
                    elif sentiment == "Negative":
                        st.error(f"**Sentiment:** {sentiment} 😔 | **Confidence:** {confidence:.2f}")
                    elif sentiment == "Neutral":
                        st.info(f"**Sentiment:** {sentiment} 😐 | **Confidence:** {confidence:.2f}")
                    else:
                        st.warning(f"**Unknown Sentiment:** {label} | **Confidence:** {confidence:.2f}")
                    
                    # Add to history
                    st.session_state.history.append({
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Review": review[:50] + "..." if len(review) > 50 else review,
                        "Sentiment": sentiment,
                        "Confidence": confidence
                    })
                        
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")

# Collapsible history section
with st.expander("📜 View Analysis History"):
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df[["Timestamp", "Review", "Sentiment", "Confidence"]].style.format({"Confidence": "{:.2f}"}))
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No analysis history yet.")


# Footer
st.markdown("---")
st.markdown("Built with Streamlit & Transformers | Model: flipkart-model | © 2025")

# # Sentiment stats with bar chart
# if st.session_state.history:
#     st.markdown("### 📈 Sentiment Insights")
#     history_df = pd.DataFrame(st.session_state.history)
    
#     # Metrics
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Total Reviews", len(history_df), help="Total number of analyzed reviews")
#     col2.metric("Positive 😊", len(history_df[history_df["Sentiment"] == "Positive"]), f"{(len(history_df[history_df['Sentiment'] == 'Positive']) / len(history_df) * 100):.1f}%")
#     col3.metric("Negative 😔", len(history_df[history_df["Sentiment"] == "Negative"]), f"{(len(history_df[history_df['Sentiment'] == 'Negative']) / len(history_df) * 100):.1f}%")
    
#     # Bar chart
#     sentiment_counts = history_df["Sentiment"].value_counts().reset_index()
#     sentiment_counts.columns = ["Sentiment", "Count"]
#     fig = px.bar(
#         sentiment_counts,
#         x="Sentiment",
#         y="Count",
#         title="Sentiment Distribution",
#         color="Sentiment",
#         color_discrete_map={"Positive": "#2874F0", "Negative": "#FB641B", "Neutral": "#FF9F00"}
#     )
#     fig.update_layout(showlegend=False, height=300)
#     st.plotly_chart(fig, use_container_width=True)
