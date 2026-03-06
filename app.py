# import streamlit as st
# import torch
# from transformers import RobertaTokenizer, RobertaForSequenceClassification
# from huggingface_hub import hf_hub_download
#
# # -----------------------------
# # Page Config
# # -----------------------------
# st.set_page_config(page_title="Emotion Detection", page_icon="😊")
# st.title("😊 Emotion Detection App")
#
# # -----------------------------
# # Emotion Labels
# # -----------------------------
# emotion_labels = ["anger", "fear", "joy", "sadness", "surprise"]
#
# NUM_LABELS = len(emotion_labels)
#
# # -----------------------------
# # Load Model
# # -----------------------------
# @st.cache_resource
# def load_model():
#
#     # Download weights from HuggingFace
#     model_path = hf_hub_download(
#         repo_id="rahul-venu/Emotion-detect",
#         filename="best_roberta_emotion.pt"
#     )
#
#     # Recreate architecture
#     model = RobertaForSequenceClassification.from_pretrained(
#         "roberta-base",
#         num_labels=NUM_LABELS
#     )
#
#     # Load state_dict
#     model.load_state_dict(
#         torch.load(model_path, map_location=torch.device("cpu"))
#     )
#
#     model.eval()
#     return model
#
# model = load_model()
#
# # -----------------------------
# # Load Tokenizer
# # -----------------------------
# @st.cache_resource
# def load_tokenizer():
#     return RobertaTokenizer.from_pretrained("roberta-base")
#
# tokenizer = load_tokenizer()
#
# # -----------------------------
# # Prediction Function
# # -----------------------------
# def predict_emotion(text):
#
#     inputs = tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         max_length=128
#     )
#
#     with torch.no_grad():
#         outputs = model(**inputs)
#
#     logits = outputs.logits
#     probs = torch.sigmoid(logits).numpy()[0]
#
#     threshold = 0.5
#
#     predicted = [
#         emotion_labels[i]
#         for i, p in enumerate(probs)
#         if p > threshold
#     ]
#
#     if not predicted:
#         return "No strong emotion detected"
#
#     return ", ".join(predicted)
#
# # -----------------------------
# # UI
# # -----------------------------
# user_input = st.text_area("Enter text")
#
# if st.button("Predict"):
#     if user_input.strip() == "":
#         st.warning("Please enter text first.")
#     else:
#         result = predict_emotion(user_input)
#         st.success(f"Predicted Emotion: {result}")



import streamlit as st
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from huggingface_hub import hf_hub_download

# -----------------------------
# CONFIG
# -----------------------------

MODEL_REPO = "rahul-venu/Emotion-detect"
MODEL_FILE = "best_roberta_emotion.pt"

LABELS = ["Anger", "Fear", "Joy", "Sadness", "Surprise"]

EMOJIS = {
    "Anger": "😡",
    "Fear": "😨",
    "Joy": "😊",
    "Sadness": "😢",
    "Surprise": "😲"
}

MAX_LEN = 128
DEVICE = "cpu"


# -----------------------------
# LOAD MODEL (cached)
# -----------------------------

@st.cache_resource
def load_model():

    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=5
    )

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE
    )

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    return model, tokenizer


model, tokenizer = load_model()


# -----------------------------
# PREDICTION FUNCTION
# -----------------------------

def predict(text):

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}

    with torch.no_grad():
        logits = model(**encoding).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    return dict(zip(LABELS, probs))


# -----------------------------
# UI
# -----------------------------

st.title("🎭 Emotion Detection")
st.write("Detect emotions in text using **RoBERTa (IITM Project)**")

text = st.text_area("Enter text")


if st.button("Analyze Emotion"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:

        probs = predict(text)

        # filter emotions above threshold
        emotions = [k for k, v in probs.items() if v > 0.5]

        if emotions:
            result = ", ".join([f"{EMOJIS[e]} {e}" for e in emotions])
        else:
            result = "No strong emotion detected"

        st.subheader("Prediction")
        st.success(result)

        st.subheader("Probabilities")

        for label, prob in probs.items():
            st.write(f"{label}: {prob:.2f}")
