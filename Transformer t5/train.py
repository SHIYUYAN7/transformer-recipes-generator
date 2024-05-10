from transformers import T5Tokenizer
import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for ingredients, steps in dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask = ingredients['input_ids'].to(device), ingredients['attention_mask'].to(device)
        labels = steps['input_ids'].to(device)
        tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
        labels[labels[:, :] == tokenizer.pad_token_id] = -100
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def calculate_perplexity(model, data_loader, device):
    model.eval()
    total_loss = 0
    total_words = 0
    with torch.no_grad():
        for ingredients, steps in data_loader:
            input_ids, attention_mask = ingredients['input_ids'].to(device), ingredients['attention_mask'].to(device)
            labels = steps['input_ids'].to(device)
            tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
            labels[labels[:, :] == tokenizer.pad_token_id] = -100
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(0)
            total_words += input_ids.size(0)
    mean_loss = total_loss / total_words
    perplexity = math.exp(mean_loss)
    return perplexity
