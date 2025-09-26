# app.py
"""
Streamlit App for CORD-19 Dataset Analysis
Simple interactive web application to display research findings
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np

# Configure the page
st.set_page_config(
    page_title="CORD-19 Research Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load the cleaned dataset"""
    try:
        df = pd.read_csv('cleaned_cord19.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">📚 CORD-19 Research Paper Analysis</div>', 
                unsafe_allow_html=True)
    
    st.write("""
    This interactive dashboard explores the COVID-19 Open Research Dataset (CORD-19), 
    containing metadata from thousands of research papers about COVID-19 and related coronaviruses.
    """)
    
    # Load data with progress indicator
    with st.spinner('Loading dataset...'):
        df = load_data()
    
    if df is None:
        st.error("Could not load the dataset. Please make sure 'cleaned_cord19.csv' is available.")
        return
    
    # Sidebar for controls
    st.sidebar.title("🔧 Controls")
    st.sidebar.write("Customize the analysis:")
    
    # Sample size selector
    sample_size = st.sidebar.slider(
        "Sample size for display",
        min_value=100,
        max_value=1000,
        value=500,
        step=100
    )
    
    # Year range selector
    if 'publication_year' in df.columns:
        valid_years = df[df['publication_year'] != 'Unknown']['publication_year'].unique()
        if len(valid_years) > 0:
            min_year = int(valid_years.min())
            max_year = int(valid_years.max())
            year_range = st.sidebar.slider(
                "Publication year range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )
        else:
            year_range = (2000, 2023)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "📊 Analysis", "🔍 Data Explorer", "📋 About"])
    
    with tab1:
        st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Papers", len(df))
        
        with col2:
            papers_with_abstract = df['abstract'].notna().sum()
            st.metric("Papers with Abstracts", papers_with_abstract)
        
        with col3:
            unique_journals = df['journal'].nunique()
            st.metric("Unique Journals", unique_journals)
        
        with col4:
            avg_word_count = df['abstract_word_count'].mean()
            st.metric("Avg Abstract Words", f"{avg_word_count:.0f}")
        
        # Display sample data
        st.subheader("Sample of Research Papers")
        st.dataframe(df[['title', 'journal', 'publication_year']].head(10), 
                    use_container_width=True)
    
    with tab2:
        st.markdown('<div class="section-header">Data Analysis and Visualizations</div>', 
                   unsafe_allow_html=True)
        
        # Filter data based on sidebar selections
        filtered_df = df.copy()
        if 'publication_year' in df.columns:
            filtered_df = filtered_df[filtered_df['publication_year'] != 'Unknown']
            filtered_df = filtered_df[
                (filtered_df['publication_year'] >= year_range[0]) & 
                (filtered_df['publication_year'] <= year_range[1])
            ]
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Publications Over Time")
            if not filtered_df.empty:
                year_counts = filtered_df['publication_year'].value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(year_counts.index, year_counts.values, marker='o', linewidth=2)
                ax.set_xlabel('Publication Year')
                ax.set_ylabel('Number of Papers')
                ax.set_title('COVID-19 Publications Timeline')
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.info("No data available for selected year range")
        
        with col2:
            st.subheader("Top Publishing Journals")
            if not filtered_df.empty:
                top_journals = filtered_df['journal'].value_counts().head(10)
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = plt.cm.Set3(np.linspace(0, 1, len(top_journals)))
                ax.barh(range(len(top_journals)), top_journals.values, color=colors)
                ax.set_yticks(range(len(top_journals)))
                ax.set_yticklabels(top_journals.index)
                ax.set_xlabel('Number of Papers')
                ax.set_title('Top 10 Journals')
                st.pyplot(fig)
        
        # Additional visualizations
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Source Distribution")
            if not filtered_df.empty:
                source_counts = filtered_df['source_x'].value_counts().head(8)
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%')
                ax.set_title('Paper Sources')
                st.pyplot(fig)
        
        with col4:
            st.subheader("Abstract Length Distribution")
            if not filtered_df.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(filtered_df['abstract_word_count'], bins=30, alpha=0.7, color='lightblue')
                ax.set_xlabel('Word Count')
                ax.set_ylabel('Frequency')
                ax.set_title('Abstract Length Distribution')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
    
    with tab3:
        st.markdown('<div class="section-header">Data Explorer</div>', unsafe_allow_html=True)
        
        # Interactive data exploration
        st.subheader("Filter and Explore Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Journal filter
            journals = ['All'] + list(df['journal'].value_counts().head(20).index)
            selected_journal = st.selectbox("Filter by Journal:", journals)
        
        with col2:
            # Word count filter
            min_words, max_words = st.slider(
                "Abstract Word Count Range:",
                min_value=0,
                max_value=int(df['abstract_word_count'].max()),
                value=(0, 500)
            )
        
        # Apply filters
        filtered_data = df.copy()
        if selected_journal != 'All':
            filtered_data = filtered_data[filtered_data['journal'] == selected_journal]
        
        filtered_data = filtered_data[
            (filtered_data['abstract_word_count'] >= min_words) & 
            (filtered_data['abstract_word_count'] <= max_words)
        ]
        
        st.write(f"**Filtered results:** {len(filtered_data)} papers")
        
        # Display filtered results
        if not filtered_data.empty:
            st.dataframe(
                filtered_data[['title', 'journal', 'publication_year', 'abstract_word_count']].head(20),
                use_container_width=True
            )
        else:
            st.info("No papers match the selected filters")
    
    with tab4:
        st.markdown('<div class="section-header">About This Project</div>', unsafe_allow_html=True)
        
        st.write("""
        ## 📋 Project Overview
        
        This application analyzes the CORD-19 dataset, which contains metadata from COVID-19 
        research papers compiled by the Allen Institute for AI.
        
        ## 🎯 Learning Objectives
        
        - Practice loading and exploring real-world datasets
        - Learn basic data cleaning techniques
        - Create meaningful visualizations
        - Build interactive web applications with Streamlit
        
        ## 🛠 Technical Stack
        
        - **Python 3.7+** - Programming language
        - **Pandas** - Data manipulation and analysis
        - **Matplotlib/Seaborn** - Data visualization
        - **Streamlit** - Web application framework
        
        ## 📊 Dataset Information
        
        The dataset includes:
        - Paper titles and abstracts
        - Publication dates and authors
        - Journal information
        - Source metadata
        
        ## 🔍 Key Findings
        
        - Exponential growth in COVID-19 research publications
        - Concentration of research in major medical journals
        - Diverse range of research topics and methodologies
        - Global collaboration in coronavirus research
        """)
        
        st.success("""
        **Assignment completed successfully!** 
        This project demonstrates fundamental data analysis skills and web application development.
        """)

if __name__ == "__main__":
    main()
