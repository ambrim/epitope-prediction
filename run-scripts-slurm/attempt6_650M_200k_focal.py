from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import evaluate
from evaluate import load
from datasets import Dataset
import pandas as pd
import ast
from sklearn.model_selection import train_test_split

model_checkpoint = "facebook/esm2_t33_650M_UR50D"

df = pd.read_csv("result_200k_1037702.csv")
print(df.shape)

def build_labels(sequence, start, end):
    # Start with all 0s
    n = len(start)
    m = len(end)
    labels = np.zeros(len(sequence), dtype=np.int8)
    if n != m:
        print(row['Epitope ID'], start, end)
        return None
    else:
        for i in range(n):
            labels[start[i]: end[i]] = 1  # 1 indicates epitope
        return labels
    

sequences = []
labels = []
to_drop = []


for index, row in df.iterrows():
    sequence = row["Epitope - Source Molecule IRI"]
    n = len(sequence)
    if n > 1022:
        to_drop.append(index)
    else:
        start_list = ast.literal_eval(row["Epitope - Starting Position"])
        end_list = ast.literal_eval(row["Epitope - Ending Position"])
        epitope_start = [int(x)-1 for x in start_list if int(x) >= 0 and int(x) <= n]
        epitope_end = [int(x) for x in end_list if int(x) >= 0 and int(x) <= n]
#       print(index, sequence)
        row_labels = build_labels(sequence, epitope_start, epitope_end)
        sequences.append(sequence)
        labels.append(row_labels)
    
df.drop(to_drop, inplace=True)

train_sequences, test_sequences, train_labels, test_labels = train_test_split(sequences, labels, test_size=0.1, shuffle=True)


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

train_tokenized = tokenizer(train_sequences)
test_tokenized = tokenizer(test_sequences)


train_dataset = Dataset.from_dict(train_tokenized)
test_dataset = Dataset.from_dict(test_tokenized)

train_dataset = train_dataset.add_column("labels", train_labels)
test_dataset = test_dataset.add_column("labels", test_labels)

num_labels = 2
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

data_collator = DataCollatorForTokenClassification(tokenizer)

model_name = model_checkpoint.split("/")[-1]
batch_size = 6

args = TrainingArguments(
    f"{model_name}-finetuned-epitope_200k_attempt6_650M_focal", # 93% 0, 7$ 1
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    weight_decay=0.001,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

# from sklearn.metrics import precision_recall_fscore_support, average_precision_score

# Load additional metrics
# precision_metric = load("precision")
# recall_metric = load("recall")
# acc_metric = load("accuracy")
seqeval = evaluate.load('seqeval')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    labels = labels.reshape((-1,))
    predictions = np.argmax(predictions, axis=2)
    predictions = predictions.reshape((-1,))
    predictions = predictions[labels!=-100]
    labels = labels[labels!=-100]

    label_map = {'LABEL_1': 1, 'LABEL_0': 0}
    predictions = [[label_map[p] for p in pred] for pred in predictions]
    references = [[label_map[l] for l in label] for label in labels]
    predictions = [pred[:len(ref)] for pred, ref in zip(predictions, references)]

    results = seqeval.compute(predictions=predictions, references=references)
    return results
#     # Calculate precision, recall, and F1 score
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')

#     # Calculate AUPRC
#     auprc = average_precision_score(labels, predictions)

#     return {
#         "accuracy": acc_metric.compute(predictions=predictions, references=labels),
#         "precision": f1_metric.compute(predictions=predictions, references=labels),
#         "recall": recall_metric.compute(predictions=predictions, references=labels),
#     }

# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs.get("logits")
        
#         # Use the weighted binary cross-entropy loss
#         loss_fct = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([14.5], device=model.device))
        
#         # Apply sigmoid to logits and reshape labels to match logits shape
#         loss = loss_fct(logits.view(-1), labels.float().view(-1))
        
#         return (loss, outputs) if return_outputs else loss

class WeightedFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = torch.nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)

        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss

        return focal_loss.mean()

# Example usage in CustomTrainer
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = WeightedFocalLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        print(labels.size(), logits.size())
        logits = logits[:, :, 1]

        loss = self.focal_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss
    
trainer = CustomTrainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)
