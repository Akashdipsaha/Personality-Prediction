import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import pdfplumber
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set up environment
sns.set_style('darkgrid')
warnings.filterwarnings('ignore')

# Function to read CSV file
def read_csv(file):
    return pd.read_csv(file)

# Function to read PDF file
def read_pdf(file):
    pdf_text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            pdf_text += page.extract_text()
    return pdf_text

# Function to preprocess text data
def cleanResume(resumeText):
    resumeText = re.sub(r'http\\S+', ' ', resumeText)  # remove URLs
    resumeText = re.sub(r'#\\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub(r'@\\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ',
                        resumeText)  # remove punctuations
    resumeText = re.sub(r'\\s+', ' ', resumeText) # remove extra whitespace
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\\w+')
    return resumeText

# Streamlit UI
st.title('Resume Data Analysis and Classification')

# File upload
uploaded_file = st.file_uploader("Upload a CSV or PDF file", type=["csv", "pdf"])

if uploaded_file:
    if uploaded_file.type == 'text/csv':
        df = read_csv(uploaded_file)
        st.write(df.head())
    elif uploaded_file.type == 'application/pdf':
        pdf_text = read_pdf(uploaded_file)
        st.text_area("PDF Text", pdf_text, height=300)
        # Create a DataFrame from PDF text if needed for further processing
        df = pd.DataFrame({'Resume': [pdf_text]})
        df['Category'] = 'Unknown'  # Placeholder category for PDF text

    # Data Cleaning
    df['cleaned'] = df['Resume'].apply(lambda x: cleanResume(x))

    # Prepare text corpus
    corpus = " ".join(df["cleaned"])

    # Generate word cloud
    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('english')
    tokenizer = nltk.tokenize.RegexpTokenizer('\w+')
    tokens = tokenizer.tokenize(corpus)
    words = [word.lower() for word in tokens if word.lower() not in stopwords]

    # Lemmatization
    nltk.download('wordnet')
    wn = WordNetLemmatizer()
    lem_words = [wn.lemmatize(word) for word in words]

    # Frequency Distribution
    freq_dist = nltk.FreqDist(lem_words)
    st.subheader('Word Frequency Distribution')

    # Plot Frequency Distribution
    fig, ax = plt.subplots(figsize=(20, 12))
    freq_words = list(freq_dist.keys())
    freq_counts = list(freq_dist.values())
    ax.bar(freq_words[:30], freq_counts[:30])  # Plot top 30 words
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.set_title('Top 30 Word Frequencies')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    # Word Cloud
    res = ' '.join([i for i in lem_words if not i.isdigit()])
    st.subheader('Word Cloud (100 Words)')
    wordcloud = WordCloud(
        background_color='black',
        max_words=100,
        width=1400,
        height=1200
    ).generate(res)
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(wordcloud)
    ax.axis('off')
    st.pyplot(fig)

    # Label Encoding
    if 'Category' in df.columns and df['Category'].nunique() > 1:  # Ensure there is more than one category
        label = LabelEncoder()
        df['new_Category'] = label.fit_transform(df['Category'])

        # Vectorizing the cleaned columns
        text = df['cleaned'].values
        target = df['new_Category'].values
        word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            stop_words='english',
            max_features=1500
        )
        word_vectorizer.fit(text)
        WordFeatures = word_vectorizer.transform(text)

        # Check if there is enough data to split
        if len(text) > 1:
            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(WordFeatures, target, random_state=24, test_size=0.2)

            # Model Training
            model = OneVsRestClassifier(KNeighborsClassifier())
            model.fit(X_train, y_train)

            # Prediction and Evaluation
            training_accuracy = model.score(X_train, y_train) * 100
            validation_accuracy = model.score(X_test, y_test) * 100

            st.subheader('Model Accuracy')
            st.write(f"Training Accuracy: {round(training_accuracy, 2)}%")
            st.write(f"Validation Accuracy: {round(validation_accuracy, 2)}%")

            # Print Classification Report
            y_pred = model.predict(X_test)
            st.text('Classification Report')
            st.text(metrics.classification_report(y_test, y_pred))
        else:
            st.warning("Not enough data to perform a train-test split. Please provide more data.")
    else:
        st.warning("No valid categories found for classification.")
