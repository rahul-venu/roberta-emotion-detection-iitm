import torch
from src.utils import get_labels_above_threshold


def predict_text(model, tokenizer, text, device, max_length, labels, threshold=0.5):
    model.eval()

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        logits = model(**encoding).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    probs_dict = dict(zip(labels, probs))


    filtered_labels = get_labels_above_threshold(probs_dict, threshold)

    return ", ".join(filtered_labels)