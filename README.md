Disease Prediction with ML and RAG Q/A

This project applies machine learning to predict diseases based on health indicators, and integrates a Retrieval-Augmented Generation (RAG) system to answer medical-related questions.

📌 Overview

Goal: Build a machine learning pipeline to detect possible diseases from patient data.

Approach: Train and evaluate multiple ML models, select the best one, and deploy it.

Extra Feature: Use RAG to provide explanations, advice, and additional medical insights.

🧠 Machine Learning Models

We experimented with several algorithms for disease prediction, including:

Logistic Regression

Decision Tree

Random Forest

Support Vector Machine (SVM)

Gradient Boosting

The best-performing model was saved as rf_model.pkl.

🔍 RAG Integration

The Retrieval-Augmented Generation system works as follows:

Retrieve: Search relevant documents from a medical knowledge base.

Augment: Combine retrieved context with the user’s question.

Generate: Produce a clear and helpful answer.

This allows the system to:

Explain predictions

Provide medical term definitions

Suggest further reading
