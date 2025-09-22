import torch
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer

# Load pretrained ESM2 model and tokenizer
model_name = "facebook/esm2_t6_8M_UR50D"
tokenizer = EsmTokenizer.from_pretrained(model_name)
esm_model = EsmModel.from_pretrained(model_name)


class ESM2WithHead(nn.Module):
    def __init__(self, esm_model, output_dim=1):
        super().__init__()
        self.esm = esm_model
        self.fc = nn.Sequential(
            nn.Linear(self.esm.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        return self.fc(cls_token)


from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

sequence = "MKTAYIAKQRQISFVKSHFSRQDILD"
inputs = tokenizer(sequence, return_tensors="pt", padding=True)

# Dummy data
input_ids = torch.randint(0, tokenizer.vocab_size, (32, 128))
attention_mask = torch.ones_like(input_ids)
labels = torch.randn(32, 1)

dataset = TensorDataset(input_ids, attention_mask, labels)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ESM2WithHead(esm_model).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
