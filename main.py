from src.model import load_model
from src.predict import predict_text
from transformers import RobertaTokenizer
from src.config import *
import torch


def main():
    model = load_model(MODEL_NAME, NUM_LABELS)
    model.load_state_dict(torch.load("models/best_roberta_emotion.pt", map_location=DEVICE))
    model.to(DEVICE)

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    text = input("Enter text: ")

    labels = predict_text(
        model=model,
        tokenizer=tokenizer,
        text=text,
        device=DEVICE,
        max_length=MAX_LEN,
        labels=LABELS,
        threshold=0.5
    )

    print("Predicted Emotion:", labels)


if __name__ == "__main__":
    main()
