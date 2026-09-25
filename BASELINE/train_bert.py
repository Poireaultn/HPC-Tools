import torch
import time
from transformers import BertTokenizerFast, BertForQuestionAnswering
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW

# 1. Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Load the requested model and tokenizer
model_name = "google-bert/bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name).to(device)

# 3. Load SQuAD
print("Loading SQuAD dataset...")
dataset = load_dataset("rajpurkar/squad", split="train[:5000]") #We only take 5000 examples

# Simplified preprocessing function for the benchmark
def preprocess_function(examples):
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        max_length=384,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=False,
    )
    # For performance benchmarking, we simulate arbitrary answer positions
    # (The GPU will perform the exact same mathematical operations as if they were perfect)
    inputs["start_positions"] = [10] * len(examples["question"])
    inputs["end_positions"] = [20] * len(examples["question"])
    return inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
tokenized_datasets.set_format("torch")

# Optimized DataLoader for HPC (num_workers and pin_memory enabled)
train_dataloader = DataLoader(
    tokenized_datasets, 
    batch_size=16, 
    shuffle=True, 
    num_workers=4,        # Uses 4 CPU cores to load data
    pin_memory=True       # Speeds up CPU -> GPU transfer
)

# 4. Optimizer
optimizer = AdamW(model.parameters(), lr=3e-5)

# 5. The timed training loop
model.train()
epochs = 2  # Set to 2 epochs. If it takes less than 1 min, you can increase to 3 or 4.

print("Starting training...")
torch.cuda.synchronize()  # Synchronize before starting the timer
t0 = time.perf_counter()

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for batch in train_dataloader:
        # Move data to GPU
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        start_positions = batch['start_positions'].to(device, non_blocking=True)
        end_positions = batch['end_positions'].to(device, non_blocking=True)

        # Standard steps seen in class
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask=attention_mask, 
                        start_positions=start_positions, end_positions=end_positions)
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()

torch.cuda.synchronize()  # Synchronize after training to stop the timer
dt = time.perf_counter() - t0

print("Training completed!")
print(f"Total GPU execution time: {dt:.2f} seconds")