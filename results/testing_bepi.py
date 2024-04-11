import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
import os
import ast
import pandas as pd

# model1_name = "attempt10_150M_200k_focal_accuracy_2"
# model_path1 = "ambrim/" + model1_name

# model1_name = "attempt11_150M_200k_focal_accuracy_2"
# path = "/scratch/gpfs/jiaweim/epitope/a-attempt11_150M_200k_focal_accuracy_2/checkpoint-130378/"

# model1_name = "attempt9_150M_200k_default_accuracy"
# path = "/scratch/gpfs/jiaweim/epitope/attempt9_150M_200k_default_accuracy/checkpoint-123516/"

# model1_name = "attempt13_150M_200k_weighted_accuracy_14"
# path = "/scratch/gpfs/jiaweim/epitope/-attempt13_150M_200k_weighted_accuracy_14/checkpoint-116654/"

# model1_name = "attempt13_150M_200k_weighted_accuracy_14_masked"
# path = "/scratch/gpfs/jiaweim/epitope/-attempt13_150M_200k_weighted_accuracy_14_masked/checkpoint-116654/"

###################################### 4/1/2023 ###################################### 

# model1_name = "attempt11_150M_200k_focal_accuracy_1"
# path = "/scratch/gpfs/jiaweim/epitope/-attempt11_150M_200k_focal_accuracy_1/checkpoint-68640/"

# model1_name = "attempt11_150M_200k_focal_accuracy_5"
# path = "/scratch/gpfs/jiaweim/epitope/-attempt11_150M_200k_focal_accuracy_5/checkpoint-126947/"

# model1_name = "attempt14_150M_200k_weighted_accuracy_14_masked"
# path = "/scratch/gpfs/jiaweim/epitope/-attempt14_150M_200k_weighted_accuracy_14_masked/checkpoint-51480/"

# model1_name = "attempt14_150M_300k_weighted_accuracy_14_masked"
# path = "/scratch/gpfs/jiaweim/epitope/-attempt14_150M_300k_weighted_accuracy_14_masked/checkpoint-64200/"

# model1_name = "attempt14_650M_300k_weighted_accuracy_14_masked"
# path = "/scratch/gpfs/jiaweim/epitope/-attempt14_650M_300k_weighted_accuracy_14_masked/checkpoint-47080"


###################################### 4/4/2023 ###################################### 

# model1_name = "attempt12_150M_200k_focal_accuracy"
# path = "/scratch/gpfs/jiaweim/epitope/-attempt12_150M_200k_focal_accuracy_2/checkpoint-85800/"


###################################### 4/5/2023 ###################################### 

# model1_name = "attempt15_150M_200k_weighted_accuracy_14_masked"
# path = "/scratch/gpfs/jiaweim/epitope/-attempt15_150M_200k_weighted_accuracy_14_masked/checkpoint-150964/"

# model1_name = "attempt10_150M_200k_default_accuracy_batches"
# path = "/scratch/gpfs/jiaweim/epitope/attempt10_150M_200k_default_accuracy_batches/checkpoint-150964/"

# model1_name = "attempt12_150M_200k_weighted_accuracy_3"
# path = "/scratch/gpfs/jiaweim/epitope/w-attempt12_150M_200k_weighted_accuracy_3/checkpoint-130378/"

###################################### 4/7/2023 ###################################### 

# model1_name = "attempt15_150M_200k_focal_weighted_accuracy_2"
# path = "/scratch/gpfs/jiaweim/epitope/attempt15_150M_200k_focal_weighted_accuracy_2/checkpoint-150964/"

# model1_name = "attempt15_150M_200k_weighted_accuracy_14_batches"
# path = "/scratch/gpfs/jiaweim/epitope/attempt15_150M_200k_weighted_accuracy_14_batches/checkpoint-150964/"

model1_name = "attempt15_150M_200k_weighted_accuracy_3_masked"
path = "/scratch/gpfs/jiaweim/epitope/-attempt15_150M_200k_weighted_accuracy_3_masked/checkpoint-130378/"

# model1_name = "attempt15_150M_200k_weighted_accuracy_5_masked"
# path = "/scratch/gpfs/jiaweim/epitope/-attempt15_150M_200k_weighted_accuracy_5_masked/checkpoint-123516/"

###################################### default 5k, 25k, 50k? ######################################

# model1_name = "default_5k"
# path = "/scratch/gpfs/jiaweim/epitope/finetuned-epitope_attempt7_150M_200k_default_accuracy-finetuned-epitope_attempt7_150M_200k_default_accuracy_more/checkpoint-18304/"

# model1_name = "default_25k"
# path = "/scratch/gpfs/jiaweim/epitope/finetuned-epitope_attempt7_150M_200k_default_accuracy_more-finetuned-epitope_attempt7_150M_200k_default_accuracy_most/checkpoint-17160/"

# model1_name = "default_50k"
# path = "/scratch/gpfs/jiaweim/epitope/finetuned-epitope_attempt7_150M_200k_default_accuracy_more-finetuned-epitope_attempt8_150M_200k_default_accuracy_mostest/checkpoint-68620/"

###################################### 4/7/2023 ###################################### 

# model1_name = "attempt13_150M_200k_focal_accuracy_2"
# path = "/scratch/gpfs/jiaweim/epitope/-attempt13_150M_200k_focal_accuracy_2/checkpoint-130378/"

###################################### 4/10/2023 ###################################### 
# model1_name = "attempt16_150M_200k_weighted_accuracy_3_masked"
# path = "/scratch/gpfs/jiaweim/epitope/-attempt16_150M_200k_weighted_accuracy_3_masked/checkpoint-61758/"

tokenizer1 = AutoTokenizer.from_pretrained(path, local_files_only=True)
model1 = AutoModelForTokenClassification.from_pretrained(path)

pipe1 = TokenClassificationPipeline(model=model1, tokenizer=tokenizer1)

mapping = {'LABEL_1': 1, 'LABEL_0': 0}

sequences = []
labels = []

count = 0
def process_fasta_file(file_path):
    with open(file_path, 'r') as file:
        fasta_data = file.read()
    count = 0

    # Split the data into different sequences
    sequences = fasta_data.strip().split('>')[1:] 
    results = []

    for seq in sequences:
        lines = seq.split('\n')
        header = lines[0].strip()
        sequence = ''.join(lines[1:]).strip()  # Join the remaining lines to get the sequence
        if len(sequence) < 1024:
            count += 1
            # Create 'true' array with 1 for uppercase and 0 for lowercase
            true_labels = []
            for char in sequence:
                if char.isupper():
                    true_labels.append(1)
                else:
                    true_labels.append(0)
            # Create 'sequence' array with all uppercase characters
            uppercase_sequence = sequence.upper()

            results.append({
                'ID': header,
                'true_labels': true_labels,
                'sequence': uppercase_sequence
            })
        
    print(count)
    
    return results

# Since I can't directly access external files, this is how you would use the function:
eval_filename = "bepi2_eval_30k"
file_path = f"{eval_filename}.fasta"
processed_fasta = process_fasta_file(file_path)

# Initialize lists to store data
true_labels, predicted_labels1, predicted_scores1, sequences = [], [], [], []
# predicted_labels2, predicted_scores2 = [], []
# predicted_labels3, predicted_scores3 = [], []
to_drop = []
ids =[]

# Initialize counters for performance metrics for both pipes
TP1, FP1, TN1, FN1 = 0, 0, 0, 0
# TP2, FP2, TN2, FN2 = 0, 0, 0, 0
# TP3, FP3, TN3, FN3 = 0, 0, 0, 0
length = len(processed_fasta)
# Processing each row in the DataFrame
for i in range(10000):
    curr = processed_fasta[i]
    id = curr.get('ID')
    ids.append(id)
    sequence = curr.get('sequence')
    sequences.append(sequence)
    # Check if sequence length is too long
    n = len(sequence)
    if n > 1022:
        to_drop.append(index)
    elif id.startswith("NegativeID"):
        to_drop.append(index)
    else:
        print(i)
        # Processing with pipe1
        result1 = pipe1(sequence)
        predicted1 = [mapping[result1[i]['entity']] for i in range(n)]
        scores1 = [result1[i]['score'] for i in range(n)]
        predicted_scores1.append(np.where(np.array(predicted1) == 0, 1 - np.array(scores1), scores1).tolist())
        true = curr.get('true_labels')
        # Processing with pipe2
#         result2 = pipe2(sequence)
#         predicted2 = [mapping[result2[i]['entity']] for i in range(n)]
#         scores2 = [result2[i]['score'] for i in range(n)]
#         predicted_scores2.append(np.where(np.array(predicted2) == 0, 1 - np.array(scores2), scores2).tolist())
#         print(index)
        
#          # Processing with pipe3
#         result3 = pipe3(sequence)
#         predicted3 = [mapping[result3[i]['entity']] for i in range(n)]
#         scores3 = [result3[i]['score'] for i in range(n)]
#         predicted_scores3.append(np.where(np.array(predicted3) == 0, 1 - np.array(scores3), scores3).tolist())
#         print(index)
        
        # Append labels to respective lists
        true_labels.append(true)
        print(true)
        predicted_labels1.append(predicted1)
#         print(predicted1)
#         predicted_labels2.append(predicted2)
#         predicted_labels3.append(predicted3)

        # Update performance metrics for both pipelines
        for tru, pred1 in zip(true_labels, predicted1):
            # Metrics for pipe1
            tru = np.array(tru)
            pred1 = np.array(pred1)
            TP1 += np.sum((tru == 1) & (pred1 == 1))
            FP1 += np.sum((tru == 0) & (pred1 == 1))
            TN1 += np.sum((tru == 0) & (pred1 == 0))
            FN1 += np.sum((tru == 1) & (pred1 == 0))
            print(TP1, TN1)
#             # Metrics for pipe2
#             TP2 += (tru == 1 and pred2 == 1)
#             FP2 += (tru == 0 and pred2 == 1)
#             TN2 += (tru == 0 and pred2 == 0)
#             FN2 += (tru == 1 and pred2 == 0)
#             # Metrics for pipe3
#             TP3 += (tru == 1 and pred3 == 1)
#             FP3 += (tru == 0 and pred3 == 1)
#             TN3 += (tru == 0 and pred3 == 0)
#             FN3 += (tru == 1 and pred3 == 0)

# Remove rows with sequences too long
# df_test.drop(to_drop, inplace=True)

# Create a DataFrame for results
results_df = pd.DataFrame({
    'Sequence': sequences,
    'ID': ids,
    'True Labels': true_labels,
    'Predicted Labels Pipe1': predicted_labels1,
    'Predicted Scores Pipe1': predicted_scores1,
    'TP Pipe1': TP1,
    'TN Pipe1': TN1,
    'FP Pipe1': FP1,
    'FN Pipe1': FN1,
#     'Predicted Labels Pipe2': predicted_labels2,
#     'Predicted Scores Pipe2': predicted_scores2,
#     'TP Pipe2': TP2,
#     'TN Pipe2': TN2,
#     'FP Pipe2': FP2,
#     'FN Pipe2': FN2,
#     'Predicted Labels Pipe3': predicted_labels3,
#     'Predicted Scores Pipe3': predicted_scores3,
#     'TP Pipe3': TP3,
#     'TN Pipe3': TN3,
#     'FP Pipe3': FP3,
#     'FN Pipe3': FN3,
})

# Save results to a JSON file
results_df.to_json(f'{eval_filename}_{model1_name}_10000.json')

print("Results saved")