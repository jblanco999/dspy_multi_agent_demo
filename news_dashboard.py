import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

# Set page config
st.set_page_config(
    page_title="News Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stPlotlyChart {
        width: 100%;
    }
    .sentiment-positive {
        color: green;
        font-weight: bold;
    }
    .sentiment-negative {
        color: red;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: gray;
        font-weight: bold;
    }
    .bias-label {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8em;
    }
    </style>
""", unsafe_allow_html=True)


def load_data():
    # In a real application, you would load your JSON data here
    # For now, we'll use sample data
    with open('output.json', 'r') as f:
        data = json.load(f)
    sample_data = data
    return pd.DataFrame(sample_data)


def get_bias_category(bias):
    if bias <= 20:
        return "Low Bias"
    elif bias <= 50:
        return "Moderate Bias"
    else:
        return "High Bias"


def main():
    st.title(f"News Analysis Dashboard: ")

    # Load data
    df = load_data()

    # Add bias category
    df['bias_category'] = df['bias'].apply(get_bias_category)

    # Sidebar filters
    st.sidebar.title("Filters")

    # Sentiment filter
    selected_sentiments = st.sidebar.multiselect(
        "Filter by Sentiment",
        options=df['sentiment'].unique(),
        default=df['sentiment'].unique()
    )

    # Bias range filter
    bias_range = st.sidebar.slider(
        "Bias Range",
        min_value=0,
        max_value=100,
        value=(0, 100)
    )

    # Apply filters
    filtered_df = df[
        (df['sentiment'].isin(selected_sentiments)) &
        (df['bias'].between(bias_range[0], bias_range[1]))
        ]

    st.subheader("Overview")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric(
            label="Total Articles",
            value=len(filtered_df)
        )

    with metric_col2:
        avg_bias = round(filtered_df['bias'].mean(), 1)
        st.metric(
            label="Average Bias Score",
            value=f"{avg_bias}%"
        )

    with metric_col3:
        sentiment_counts = filtered_df['sentiment'].value_counts()
        dominant_sentiment = sentiment_counts.index[0] if not sentiment_counts.empty else "N/A"
        st.metric(
            label="Dominant Sentiment",
            value=dominant_sentiment,
            delta=f"{round((sentiment_counts[dominant_sentiment] / len(filtered_df)) * 100)}% of articles" if not sentiment_counts.empty else "N/A"
        )

    with metric_col4:
        bias_categories = filtered_df['bias_category'].value_counts()
        most_common_bias = bias_categories.index[0] if not bias_categories.empty else "N/A"
        st.metric(
            label="Most Common Bias Category",
            value=most_common_bias,
            delta=f"{round((bias_categories[most_common_bias] / len(filtered_df)) * 100)}% of articles" if not bias_categories.empty else "N/A"
        )

    # Add a divider
    st.markdown("---")

    st.subheader("Tl;Dr")
    # summary_col1 = st.columns(1)
    with open('output_assessment.txt', 'r') as f:
        content = f.read()
    st.text_area(
        label="General Summary",
        value=content,
        height=500
    )

    # Create two columns for the main content
    col1, col2 = st.columns([2, 1])

    with col1:
        # Bias Distribution Chart
        st.subheader("Bias Distribution")
        fig_bias = px.histogram(
            filtered_df,
            x='bias',
            nbins=20,
            title="Distribution of Bias Scores",
            labels={'bias': 'Bias Score', 'count': 'Number of Articles'}
        )
        st.plotly_chart(fig_bias, use_container_width=True)

        # Sentiment Distribution
        st.subheader("Sentiment Distribution")
        fig_sentiment = px.pie(
            filtered_df,
            names='sentiment',
            title="Article Sentiment Distribution"
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)

    with col2:
        # Bias Category Distribution
        st.subheader("Bias Categories")
        fig_categories = px.bar(
            filtered_df['bias_category'].value_counts().reset_index(),
            x='count',
            y='bias_category',
            title="Distribution of Bias Categories",
            labels={'index': 'Category', 'bias_category': 'Count'}
        )
        st.plotly_chart(fig_categories, use_container_width=True)

    # Article List
    st.subheader("Article List")
    sort_order = st.selectbox(
        "Sort articles",
        options=["A to Z ↑", "Z to A ↓"],
        index=0
    )

    # Sort the dataframe based on selection
    if sort_order == "A to Z ↑":
        filtered_df = filtered_df.sort_values('title', ascending=True)
    else:
        filtered_df = filtered_df.sort_values('title', ascending=False)
    for _, article in filtered_df.iterrows():
        with st.expander(article['title']):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**Summary:**\n{article['summary_and_assessment']}")

            with col2:
                st.markdown(
                    f"**Sentiment:** <span class='sentiment-{article['sentiment'].lower()}'>"
                    f"{article['sentiment']}</span>",
                    unsafe_allow_html=True
                )
                st.markdown(f"**Bias Score:** {article['bias']}")
                st.markdown(f"**Bias Category:** {article['bias_category']}")
                if article['url']:
                    st.markdown(f"[Read Full Article]({article['url']})")


if __name__ == "__main__":
    main()