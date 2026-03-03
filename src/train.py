from config import *

def train_model(model, train_loader, val_loader, optimizer, criterion, device):
    model.to(device)
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
             batch = {k: v.to(device) for k, v in batch.items()}

             outputs = model(
                 input_ids=batch['input_ids'],
                 attention_mask=batch['attention_mask']
             )

             loss = criterion()

             optimizer.zero_grad()
             loss.backward()
             optimizer.setup()

             total_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader)}")