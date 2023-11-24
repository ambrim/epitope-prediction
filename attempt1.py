import os
import numpy as np
import pandas as pd
# from evaluate import load

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification

from sklearn.metrics import f1_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

access_token = os.environ.get("HF_HOME_TOKEN")

from huggingface_hub import login

login(token='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', write_permission=True)

# model_checkpoint = "facebook/esm2_t36_3B_UR50D"

# df = pd.read_csv("processed2.csv")

model_checkpoint = "facebook/esm2_t12_35M_UR50D"

df = pd.read_csv("processed1.csv")
df.head()

def build_labels(sequence, start, end):
    # Start with all 0s
    labels = np.zeros(len(sequence), dtype=np.int64)
    labels[epitope_start: epitope_end] = 1  # 1 indicates epitope
   
    return labels

sequences = []
labels = []
print(df.shape)

for index, row in df.iterrows():
    sequence = row["Epitope - Source Molecule IRI"]
    epitope_start = int(row["Epitope - Starting Position"]) - 1
    epitope_end = int(row["Epitope - Ending Position"])
    print(index)
    if epitope_end >= len(sequence) or epitope_start <= 0:
        df.drop(index, inplace=True)
    row_labels = build_labels(sequence, epitope_start, epitope_end)
    sequences.append(row["Epitope - Source Molecule IRI"])
    labels.append(row_labels)

train_sequences, test_sequences, train_labels, test_labels = train_test_split(sequences, labels, test_size=0.15, shuffle=True)



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
batch_size = 8

args = TrainingArguments(
    f"{model_name}-finetuned-epitope",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.001,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
)


# # metric = load("accuracy")


# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     labels = labels.reshape((-1,))
#     predictions = np.argmax(predictions, axis=2)
#     predictions = predictions.reshape((-1,))
#     predictions = predictions[labels!=-100]
#     labels = labels[labels!=-100]

#     # Compute F1 Score
#     f1 = f1_score(labels, predictions, average='weighted')
#     return {"accuracy": metric.compute(predictions=predictions, references=labels), "f1": f1}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    labels = labels.reshape((-1,))
    predictions = np.argmax(predictions, axis=2)
    predictions = predictions.reshape((-1,))
    
    # Filter out special tokens
    valid_indices = labels != -100
    predictions = predictions[valid_indices]
    labels = labels[valid_indices]

    # Compute F1 Score
    f1 = f1_score(labels, predictions, average='weighted')

    # Compute Accuracy
    accuracy = np.mean(predictions == labels)

    return {"accuracy": accuracy, "f1": f1}

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

trainer.train()

def plot_precision_recall_curve(labels, predictions):
    precision, recall, _ = precision_recall_curve(labels, predictions)
    auc_score = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, label=f'Precision-Recall curve (area = {auc_score:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.show()

