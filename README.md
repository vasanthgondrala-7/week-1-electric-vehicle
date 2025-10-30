# WEEK-1-PROJECT-ON-ELECTRIC-VEHICLES

 🚗 Chatbot on Electric Vehicles (EVs)

 📘 Project Overview

This project implements a **chatbot** trained on the **One Electric Vehicle (EV) Dataset (Smaller)** to provide informative, conversational responses about electric vehicles.
The chatbot uses natural language processing (NLP) techniques to understand queries and return relevant information on EV specifications, charging infrastructure, environmental impact, and more.

---

⚙️ Process Workflow

1. Dataset Collection

* Dataset: *One Electric Vehicle Dataset – Smaller Version*
* Contains summarized EV information such as:

  * Vehicle model names
  * Battery capacity
  * Range per charge
  * Charging time
  * Price, efficiency, and brand details

2. Data Preprocessing

* Cleaned missing and inconsistent entries
* Normalized textual data (lowercasing, punctuation removal)
* Removed irrelevant or duplicate data points
* Tokenized text for language model training

3. Chatbot Design

* Implemented using either:

  * **Rule-based approach** (intent classification + response templates), or
  * **Retrieval-based model** (semantic search across dataset)
* Integrated a simple NLP pipeline for:

  * Intent detection
  * Entity recognition (EV model, specs, etc.)
  * Response generation

4. Model Training

* Used **TF-IDF + Cosine Similarity** or **Transformer embeddings** (e.g., BERT/Sentence-BERT) for query-response mapping
* Fine-tuned or indexed dataset text to enable quick semantic retrieval

5. Chatbot Interface

* Built a minimal **Streamlit** or **Gradio** web interface for interaction
* User can input queries like:

  > “What is the range of Tata Nexon EV?”
  > “Which EV charges fastest?”
* Bot responds with data-driven answers from the dataset

6. Evaluation

* Evaluated using:

  * Response relevance score (manual check or semantic similarity)
  * User satisfaction feedback during testing

#### **7. Deployment**

* Hosted on a cloud service (e.g., Streamlit Cloud, Hugging Face Spaces, or Flask app)
* Dataset and model stored locally or on a lightweight public repository

---

🧩 Tools & Libraries

Python, Pandas, NumPy

---

 📦 Repository Structure

```
EV-Chatbot/
│
├── data/
│   └── one_ev_dataset_smaller.csv
│
├── notebooks/
│   └── data_preprocessing.ipynb
│
├── src/
│   ├── chatbot.py
│   ├── model.py
│   └── utils.py
│
├── app.py               # Main Streamlit/Gradio app
├── requirements.txt
└── README.md
```

---

#🚀 Future Enhancements

* Integrate a **Generative AI model** for open-ended answers
* Add **multilingual query support**
* Expand dataset with live EV updates from APIs

