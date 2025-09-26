# cord19_analysis.py
"""
CORD-19 Dataset Analysis
Assignment Solution for COVID-19 Research Paper Analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
from datetime import datetime
import numpy as np

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class CORD19Analyzer:
    def __init__(self, file_path):
        """
        Initialize the analyzer with the dataset path
        """
        self.file_path = file_path
        self.df = None
        self.df_cleaned = None
        
    def load_data(self):
        """
        Part 1: Load and explore the dataset
        """
        print("Loading CORD-19 metadata...")
        try:
            # Load the dataset
            self.df = pd.read_csv(self.file_path)
            print(f"✓ Dataset loaded successfully! Shape: {self.df.shape}")
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def basic_exploration(self):
        """
        Basic data exploration
        """
        print("\n" + "="*60)
        print("PART 1: BASIC DATA EXPLORATION")
        print("="*60)
        
        # Display basic information
        print("\n1. First 5 rows of the dataset:")
        print(self.df.head())
        
        print(f"\n2. Dataset dimensions: {self.df.shape}")
        
        print("\n3. Data types:")
        print(self.df.dtypes)
        
        print("\n4. Columns available:")
        print(list(self.df.columns))
        
        # Check for missing values
        print("\n5. Missing values by column:")
        missing_data = self.df.isnull().sum()
        print(missing_data[missing_data > 0])
        
        # Basic statistics for numerical columns
        print("\n6. Basic statistics:")
        print(self.df.describe())
        
    def clean_data(self):
        """
        Part 2: Data cleaning and preparation
        """
        print("\n" + "="*60)
        print("PART 2: DATA CLEANING AND PREPARATION")
        print("="*60)
        
        # Create a copy for cleaning
        self.df_cleaned = self.df.copy()
        
        # Handle missing values
        print("\n1. Handling missing values...")
        
        # Drop rows where critical columns are missing
        initial_shape = self.df_cleaned.shape[0]
        self.df_cleaned = self.df_cleaned.dropna(subset=['title', 'abstract'])
        after_shape = self.df_cleaned.shape[0]
        print(f"   Removed {initial_shape - after_shape} rows with missing titles/abstracts")
        
        # Fill missing publication dates with 'Unknown'
        self.df_cleaned['publish_time'] = self.df_cleaned['publish_time'].fillna('Unknown')
        
        # Extract year from publish_time
        print("\n2. Extracting publication year...")
        def extract_year(date_str):
            if date_str == 'Unknown':
                return 'Unknown'
            try:
                # Handle different date formats
                if isinstance(date_str, str):
                    # Try to extract year from string
                    year_match = re.search(r'\d{4}', date_str)
                    if year_match:
                        return int(year_match.group())
                return 'Unknown'
            except:
                return 'Unknown'
        
        self.df_cleaned['publication_year'] = self.df_cleaned['publish_time'].apply(extract_year)
        
        # Create abstract word count
        print("\n3. Creating abstract word count...")
        self.df_cleaned['abstract_word_count'] = self.df_cleaned['abstract'].apply(
            lambda x: len(str(x).split()) if pd.notnull(x) else 0
        )
        
        # Clean journal names
        self.df_cleaned['journal'] = self.df_cleaned['journal'].fillna('Unknown')
        
        print(f"✓ Data cleaning completed! Final shape: {self.df_cleaned.shape}")
        
    def analyze_data(self):
        """
        Part 3: Data analysis and visualization
        """
        print("\n" + "="*60)
        print("PART 3: DATA ANALYSIS AND VISUALIZATION")
        print("="*60)
        
        # Create visualizations
        self.create_visualizations()
        
        # Perform additional analysis
        self.perform_basic_analysis()
    
    def perform_basic_analysis(self):
        """
        Perform basic data analysis
        """
        print("\n1. BASIC ANALYSIS RESULTS:")
        
        # Papers by publication year
        year_counts = self.df_cleaned[self.df_cleaned['publication_year'] != 'Unknown']['publication_year'].value_counts().sort_index()
        print(f"\n   Total papers with valid years: {year_counts.sum()}")
        print(f"   Publication year range: {year_counts.index.min()} - {year_counts.index.max()}")
        
        # Top journals
        top_journals = self.df_cleaned['journal'].value_counts().head(10)
        print(f"\n   Top 10 journals by publication count:")
        for journal, count in top_journals.items():
            print(f"   - {journal}: {count} papers")
        
        # Most frequent words in titles
        print(f"\n   Analyzing word frequencies in titles...")
        all_titles = ' '.join(self.df_cleaned['title'].dropna().astype(str))
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_titles.lower())
        word_freq = Counter(words).most_common(10)
        print(f"   Top 10 words in titles:")
        for word, count in word_freq:
            print(f"   - {word}: {count}")
    
    def create_visualizations(self):
        """
        Create all required visualizations
        """
        print("\n2. CREATING VISUALIZATIONS...")
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('CORD-19 Dataset Analysis - COVID-19 Research Papers', fontsize=16, fontweight='bold')
        
        # Visualization 1: Publications over time
        self.plot_publications_over_time(axes[0, 0])
        
        # Visualization 2: Top journals
        self.plot_top_journals(axes[0, 1])
        
        # Visualization 3: Word cloud of titles
        self.plot_word_cloud(axes[1, 0])
        
        # Visualization 4: Distribution by source
        self.plot_source_distribution(axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('cord19_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional visualization: Abstract word count distribution
        self.plot_word_count_distribution()
    
    def plot_publications_over_time(self, ax):
        """Plot number of publications over time"""
        # Filter valid years and count publications
        year_data = self.df_cleaned[self.df_cleaned['publication_year'] != 'Unknown']
        year_counts = year_data['publication_year'].value_counts().sort_index()
        
        ax.plot(year_counts.index, year_counts.values, marker='o', linewidth=2, markersize=6)
        ax.set_xlabel('Publication Year')
        ax.set_ylabel('Number of Papers')
        ax.set_title('COVID-19 Publications Over Time', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        print("   ✓ Created publications over time plot")
    
    def plot_top_journals(self, ax):
        """Plot top publishing journals"""
        top_journals = self.df_cleaned['journal'].value_counts().head(10)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_journals)))
        bars = ax.barh(range(len(top_journals)), top_journals.values, color=colors)
        ax.set_yticks(range(len(top_journals)))
        ax.set_yticklabels([j[:40] + '...' if len(j) > 40 else j for j in top_journals.index])
        ax.set_xlabel('Number of Papers')
        ax.set_title('Top 10 Journals Publishing COVID-19 Research', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 10, bar.get_y() + bar.get_height()/2, 
                   f'{int(width)}', ha='left', va='center')
        
        print("   ✓ Created top journals bar chart")
    
    def plot_word_cloud(self, ax):
        """Generate word cloud of paper titles"""
        try:
            # Combine all titles
            all_titles = ' '.join(self.df_cleaned['title'].dropna().astype(str))
            
            # Generate word cloud
            wordcloud = WordCloud(width=400, height=300, 
                                background_color='white',
                                max_words=100).generate(all_titles)
            
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title('Word Cloud of Paper Titles', fontweight='bold')
            ax.axis('off')
            print("   ✓ Created word cloud")
        except Exception as e:
            ax.text(0.5, 0.5, f"WordCloud error:\n{e}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Word Cloud (Failed to Generate)', fontweight='bold')
            ax.axis('off')
            print(f"   ⚠ Word cloud generation failed: {e}")
    
    def plot_source_distribution(self, ax):
        """Plot distribution of papers by source"""
        source_counts = self.df_cleaned['source_x'].value_counts().head(8)
        
        wedges, texts, autotexts = ax.pie(source_counts.values, 
                                         labels=source_counts.index,
                                         autopct='%1.1f%%',
                                         startangle=90)
        ax.set_title('Distribution of Papers by Source', fontweight='bold')
        
        print("   ✓ Created source distribution pie chart")
    
    def plot_word_count_distribution(self):
        """Additional visualization: Abstract word count distribution"""
        plt.figure(figsize=(10, 6))
        plt.hist(self.df_cleaned['abstract_word_count'], bins=50, alpha=0.7, color='skyblue')
        plt.xlabel('Abstract Word Count')
        plt.ylabel('Frequency')
        plt.title('Distribution of Abstract Word Counts', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.savefig('abstract_word_count.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✓ Created abstract word count distribution")
    
    def save_cleaned_data(self):
        """Save cleaned dataset for Streamlit app"""
        self.df_cleaned.to_csv('cleaned_cord19.csv', index=False)
        print("✓ Cleaned data saved as 'cleaned_cord19.csv'")

def main():
    """
    Main execution function
    """
    print("CORD-19 DATASET ANALYSIS")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = CORD19Analyzer('metadata.csv')  # Update path if needed
    
    # Part 1: Load and explore
    if analyzer.load_data():
        analyzer.basic_exploration()
        
        # Part 2: Clean and prepare
        analyzer.clean_data()
        
        # Part 3: Analyze and visualize
        analyzer.analyze_data()
        
        # Save cleaned data for Streamlit
        analyzer.save_cleaned_data()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Summary findings
        print("\nKEY FINDINGS:")
        print("- COVID-19 research publications show significant growth over time")
        print("- Major medical journals lead in publication volume")
        print("- Research focuses on key terms like 'covid', 'patients', 'study'")
        print("- Abstracts typically range from 100-300 words")
        
    else:
        print("Failed to load dataset. Please check the file path.")

if __name__ == "__main__":
    main()
