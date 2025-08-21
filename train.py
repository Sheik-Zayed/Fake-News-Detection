from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

# Load and prepare dataset
df = pd.read_csv("Final Dataset.csv")
df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2)

# Tokenize
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
train_enc = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
val_enc = tokenizer(list(val_texts), truncation=True, padding=True, max_length=512)

class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} | {'labels': torch.tensor(self.labels[idx])}

    def __len__(self):
        return len(self.labels)

train_dataset = FakeNewsDataset(train_enc, list(train_labels))
val_dataset = FakeNewsDataset(val_enc, list(val_labels))

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Training setup
training_args = TrainingArguments(
    output_dir="./bert_fake_news_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train and save
trainer.train()
model.save_pretrained("bert_fake_news_model")
tokenizer.save_pretrained("bert_fake_news_model")
