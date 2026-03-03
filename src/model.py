from transformers import RobertaForSequenceClassification

def load_model(model_name, num_labels):
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    return model