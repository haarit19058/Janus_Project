
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd

class SMILESDataset(Dataset):
    def __init__(self, smiles, labels, tokenizer):
        self.smiles = smiles
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        # Access the SMILES string using .iloc to handle custom indices:
        encoding = self.tokenizer(self.smiles.iloc[idx], return_tensors="pt", padding='max_length', truncation=True, max_length=512) 
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return {**encoding, 'labels': label}

# Load your data
data = pd.read_csv("data.csv")  # Replace with your actual data file
smiles_list = data['SMILES']
s1_t1 = (1/(1+data['S1-T1'])).tolist()

# Split the dataset (80% train, 20% validation)
train_size = int(0.8 * len(smiles_list))
train_smiles = smiles_list[:train_size]
val_smiles = smiles_list[train_size:]
train_labels = s1_t1[:train_size]
val_labels = s1_t1[train_size:]

# Load the tokenizer and model
model_name = "seyonec/ChemBERTa-zinc-base-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

# Prepare datasets and dataloaders
train_dataset = SMILESDataset(train_smiles, train_labels, tokenizer)
val_dataset = SMILESDataset(val_smiles, val_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
for epoch in range(3):  # Adjust epochs as needed
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].squeeze(1).to(device)
        attention_mask = batch['attention_mask'].squeeze(1).to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Validation (optional)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

    print(f"Epoch {epoch + 1}: Validation Loss: {total_loss / len(val_loader)}")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")
