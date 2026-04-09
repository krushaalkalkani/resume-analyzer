---
title: Smart Resume Analyzer
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
---

# Smart Resume Analyzer

An ML-powered web app that predicts job categories from resumes and performs skill gap analysis.

## Features

- Upload a PDF resume or paste text
- Predicts job category using a trained Linear SVC model
- Extracts skills mentioned in the resume
- Shows missing skills for the predicted category

## Tech Stack

- Python, scikit-learn, pandas
- TF-IDF vectorization for text features
- Streamlit for the web interface
- pdfplumber for PDF text extraction

## Model

Trained on 2,484 resumes across 24 job categories with 72.4% test accuracy using Linear SVC.

Built by Krushal Kalkani.
