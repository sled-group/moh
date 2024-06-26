from collections import Counter

def precision_score(y_true, y_pred, average='macro'):
    if average not in ['macro', 'micro', 'weighted']:
        raise ValueError("Unsupported average type. Supported types: 'macro', 'micro', 'weighted'")
    
    true_positive = Counter()
    predicted_positive = Counter()
    actual_positive = Counter()

    for t, p in zip(y_true, y_pred):
        if t == p:
            true_positive[t] += 1
        predicted_positive[p] += 1
        actual_positive[t] += 1

    if average == 'micro':
        total_true_positive = sum(true_positive.values())
        total_predicted_positive = sum(predicted_positive.values())
        return total_true_positive / total_predicted_positive if total_predicted_positive > 0 else 0

    precision_per_class = {label: true_positive[label] / predicted_positive[label] if predicted_positive[label] > 0 else 0 for label in predicted_positive}

    if average == 'macro':
        return sum(precision_per_class.values()) / len(precision_per_class)
    elif average == 'weighted':
        total = sum(actual_positive.values())
        return sum(precision_per_class[label] * actual_positive[label] for label in precision_per_class) / total

def recall_score(y_true, y_pred, average='macro'):
    if average not in ['macro', 'micro', 'weighted']:
        raise ValueError("Unsupported average type. Supported types: 'macro', 'micro', 'weighted'")
    
    true_positive = Counter()
    actual_positive = Counter()

    for t, p in zip(y_true, y_pred):
        if t == p:
            true_positive[t] += 1
        actual_positive[t] += 1

    if average == 'micro':
        total_true_positive = sum(true_positive.values())
        total_actual_positive = sum(actual_positive.values())
        return total_true_positive / total_actual_positive if total_actual_positive > 0 else 0

    recall_per_class = {label: true_positive[label] / actual_positive[label] if actual_positive[label] > 0 else 0 for label in actual_positive}

    if average == 'macro':
        return sum(recall_per_class.values()) / len(recall_per_class)
    elif average == 'weighted':
        total = sum(actual_positive.values())
        return sum(recall_per_class[label] * actual_positive[label] for label in recall_per_class) / total

def f1_score(y_true, y_pred, average='macro'):
    if average not in ['macro', 'micro', 'weighted']:
        raise ValueError("Unsupported average type. Supported types: 'macro', 'micro', 'weighted'")
    
    precision = precision_score(y_true, y_pred, average)
    recall = recall_score(y_true, y_pred, average)

    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)