# Resume Data Analysis and Classification

## Overview

This project is a resume data analysis and classification application built with Streamlit. It allows users to upload CSV or PDF files containing resumes, preprocess the text data, visualize word frequency distributions, generate word clouds, and train a machine learning model for resume classification.

## Features

- **File Upload:** Support for CSV and PDF file uploads.
- **Data Cleaning:** Processes and cleans text data from resumes.
- **Text Analysis:**
  - **Word Frequency Distribution:** Displays a bar chart of the most common words.
  - **Word Cloud:** Generates and displays a word cloud from the resume text.
- **Classification:** Trains a machine learning model to classify resumes based on pre-defined categories.
- **Model Evaluation:** Provides accuracy metrics and a classification report.

## Requirements

- Python 3.x
- Streamlit
- pandas
- numpy
- matplotlib
- seaborn
- nltk
- wordcloud
- scikit-learn
- pdfplumber

Install the necessary packages using pip:

```bash
pip install streamlit pandas numpy matplotlib seaborn nltk wordcloud scikit-learn pdfplumber
