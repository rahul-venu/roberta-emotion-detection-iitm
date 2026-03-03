import numpy as np
import torch

def evaluate(model, loader, device):
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            ).logits

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    return np.vstack(all_probs)
