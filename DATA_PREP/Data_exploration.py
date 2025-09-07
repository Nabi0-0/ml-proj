import pandas as pd
import matplotlib.pyplot as plt

# Load the data
train = pd.read_csv('train.csv')

# Basic info
print("Dataset shape:", train.shape)
print("\nLabel distribution:")
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
for col in label_cols:
    print(f"{col}: {train[col].sum()} ({train[col].mean()*100:.2f}%)")

# Look at first few toxic vs clean comments
print("\n=== TOXIC EXAMPLES ===")
toxic_examples = train[train['toxic']==1]['comment_text'].head(3)
for i, comment in enumerate(toxic_examples):
    print(f"{i+1}. {comment[:100]}...")

print("\n=== CLEAN EXAMPLES ===")  
clean_examples = train[train['toxic']==0]['comment_text'].head(3)
for i, comment in enumerate(clean_examples):
    print(f"{i+1}. {comment[:100]}...")
