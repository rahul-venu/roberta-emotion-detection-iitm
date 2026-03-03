def get_labels_above_threshold(probs_dict, threshold=0.5):
    return [
        label for label, prob in probs_dict.items()
        if prob > threshold
    ]
