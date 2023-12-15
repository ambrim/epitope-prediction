import pandas as pd
import openpyxl
import requests as r
from Bio import SeqIO
from io import StringIO
from Bio import Entrez

raw_excel = 'epitope_table_large.xlsx'

# Use pandas.read_excel() to read the Excel file into a DataFrame
df = pd.read_excel(raw_excel)

df = df.dropna(subset=['Epitope - Starting Position', 'Epitope - Ending Position', 'Epitope - Source Molecule IRI'])

valid_prefixes = ['http://www.ncbi.nlm.nih.gov/protein/', 'https://www.uniprot.org/uniprot/']

# Create a boolean mask to filter rows that start with a valid prefix
mask = df['Epitope - Source Molecule IRI'].str.startswith(tuple(valid_prefixes))

# Use the boolean mask to filter the DataFrame
df = df[mask]
# only the first 20,000 rows
df = df.iloc[:50000]

Entrez.email = "iamambri@gmail.com"  # Set your email address
df['Epitope ID'] = 0

# Function to retrieve sequence from URL
def get_sequence_from_url(prefix, cID):
    try:  
        if prefix == "http://www.ncbi.nlm.nih.gov/protein/":
            handle = Entrez.efetch(db="protein", id=cID,  rettype="fasta")
            fasta_sequence = handle.read()
            handle.close()
            # Skip the first header line
            lines = fasta_sequence.split('\n')
            sequence_lines = lines[1:]
            sequence_text = ''.join(sequence_lines)
            if len(sequence_text) > 1024:
                    return 'TOO LONG'
            return sequence_text
        elif prefix == "https://www.uniprot.org/uniprot/":
            searchUrl ='http://www.uniprot.org/uniprot/' + cID + ".fasta"
        
            response = r.get(searchUrl)
        
            if response.status_code == 200:
                fasta_sequence = response.text
                # Skip the first header line
                lines = fasta_sequence.split('\n')
                sequence_lines = lines[1:]
                sequence_text = ''.join(sequence_lines)
                if len(sequence_text) > 1024:
                    return 'TOO LONG'
                return sequence_text
            else:
                print("Failed to retrieve the FASTA sequence.")
    except Exception as e:
        print(f"Error fetching data for {prefix}{cID}: {e}")

# Iterate through the filtered DataFrame
for index, row in df.iterrows():
    url = row['Epitope - Source Molecule IRI']
    
    # Check if the URL starts with one of the valid prefixes
    for prefix in valid_prefixes:
        if url[:len(prefix)] == prefix:
            # Extract the code snippet from the URL
            cID = url[len(prefix):]
           
            sequence = get_sequence_from_url(prefix, cID)
        
            if sequence == 'TOO LONG':
                df.drop(index, inplace=True)
            else:
                # Replace the URL in 'Col2' with the retrieved sequence
                df.at[index, 'Epitope - Source Molecule IRI'] = sequence
                df.at[index, 'Epitope ID'] = cID
            break
    print(index, sequence)

df = df.dropna(subset=['Epitope ID'])
file_path = 'processed_50k.csv'
df.to_csv(file_path) 
