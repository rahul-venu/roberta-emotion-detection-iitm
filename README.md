# 🎭 Emotion Detection App (RoBERTa + Streamlit)
Overview

This project implements a multi-label emotion classification system using a fine-tuned RoBERTa transformer model and a Streamlit web interface.

The model predicts one or more emotions from user input text using threshold-based filtering.

# 🧠 Model Architecture

* Base Model: roberta-base

* Framework: PyTorch

* Task: Multi-label emotion classification

* Activation: Sigmoid

* Loss Function: BCEWithLogitsLoss

* Output: Emotions with probability > 0.5

Example:
```
Input: I'm feeling overwhelmed today.
Output: Fear, Sadness
```
# 🌐 Live App (Streamlit)
```
https://text-emotion-finder.streamlit.app/
```
# 🏗 Project Structure

```
emotion-detection-project/
│
├── data/                         # Dataset files
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── submission.csv
│
├── notebook/                     # Jupyter notebooks
│   └── emotion_dtect.ipynb
│
├── src/                          # Source code (modular pipeline)
│   ├── __init__.py
│   ├── config.py                 # Configuration variables
│   ├── dataset.py                # Data loading & preprocessing
│   ├── model.py                  # Model architecture
│   ├── train.py                  # Training logic
│   ├── evaluate.py               # Evaluation metrics
│   ├── predict.py                # Inference pipeline
│   └── utils.py                  # Helper functions
|
├── app/                          
│   └── streamlit_app.py          # Frontend layer
|
├── main.py                       # Entry point to run pipeline
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```
# ☁️ Model Storage

The trained model weights are hosted externally (cloud storage) to keep the Git repository lightweight.
HuggingFace Hub 🤗

```
https://huggingface.co/rahul-venu/Emotion-detect
```

The model is downloaded at runtime inside the Streamlit app.

# ⚙️ Installation (Local Setup)
1️⃣ Clone Repository
```
git clone <your-repo-link>
cd GenAI Project-(IITM)
```

2️⃣ Create Virtual Environment
```
python -m venv .venv
.venv\Scripts\activate   # Windows
```

3️⃣ Install Dependencies
```
pip install -r requirements.txt
```

▶️ Run CLI Version
```
python main.py
```

# 🚀 Run Streamlit App
```
streamlit run app.py
```
# 💻 Streamlit App Features

* Text input box

* Real-time emotion prediction

* Clean formatted output

* Multi-label emotion display

# 📦 Requirements

* Main libraries used:

* torch

* transformers

* streamlit

* numpy

* pandas

* tokenizers

See requirements.txt for exact versions.

# 🌍 Deployment
* Streamlit Cloud Deployment Steps

* Push code to GitHub

* Go to https://streamlit.io/cloud

* Connect GitHub repo

* Select app.py

* Deploy

* Ensure requirements.txt is present.

# 🧪 How Prediction Works

* Input text is tokenized

* Model generates logits

* Sigmoid converts logits to probabilities

* Emotions above threshold are selected

* Output formatted as comma-separated string

# 📈 Future Improvements

* Display confidence scores

* Add emotion probability bars

* Add REST API endpoint

* Dockerize application

* Deploy on HuggingFace Spaces

# 💼 Technical Highlights

* Transformer fine-tuning

* Multi-label classification

* Clean modular architecture

* Threshold-based inference

* Streamlit deployment ready

* Cloud-based model hosting









