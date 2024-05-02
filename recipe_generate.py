import torch
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from MyLSTM import MyLSTM
import torch.nn as nn
    

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)


def generate_recipe(model, tokenizer, ingredients, max_length=128):
    model.eval()  # Switch to evaluation mode

    # Process ingredients into a single string if list, and tokenize
    ingredients = ' '.join(ingredients) if isinstance(ingredients, list) else ingredients
    input_ids = tokenizer.encode(ingredients, return_tensors='pt')

    # Prepare initial hidden state and cell state for LSTM
    # Initialize for all layers: num_layers, batch_size, hidden_dim
    h, c = [torch.zeros(model.lstm.num_layers, 1, model.lstm.hidden_size) for _ in range(2)]  # Adjusted for all layers

    # Prepare the sequence for generated ids, starting with the initial input
    generated_ids = input_ids

    with torch.no_grad():
        for _ in range(max_length):
            output, (h, c) = model.lstm(model.embedding(generated_ids[:, -1:]), (h, c))
            output = model.fc(output.squeeze(1))  # Assuming model.fc is the final fully connected layer
            next_token_id = torch.argmax(output, dim=1).unsqueeze(1)

            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

            # Check if the next token is the separator token, indicating the end
            if next_token_id.item() == tokenizer.sep_token_id:
                break

    # Decode the sequence of token ids to a string
    generated_text = tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True)
    print('Generated IDs:', generated_ids.squeeze())
    return generated_text


if __name__ == "__main__":
    # data preprocessing
    data = pd.read_excel("recipes_20000.xlsx")
    training_data = data.head(1000)
    max_length = 128

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size

    input_ids = []
    output_ids = []

    for index, row in training_data.iterrows():
        # string (list shape) to string(space split)
        ingredients = (' ').join(eval(row['ingredients']))
        steps = (' ').join(eval(row['steps']))

        # tokenized input and output
        input_tokens = tokenizer(ingredients, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
        output_tokens = tokenizer(steps, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)

        # add label
        input_ids.append(input_tokens['input_ids'])
        output_ids.append(output_tokens['input_ids'])

        # convert to torch
        input_ids_tensor = torch.stack(input_ids)
        output_ids_tensor = torch.stack(output_ids)

    # Training for MyLSTM
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 5
    dropout_rate = 0.1
    batch_size = 64
    epochs = 5

    # Model
    model_mylstm = MyLSTM(units=vocab_size, input_size=embedding_dim)

    dataset = TensorDataset(input_ids_tensor, output_ids_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_mylstm.parameters(), lr=0.001)

    model_mylstm.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.float()  
            targets = targets.squeeze(1) 

            optimizer.zero_grad()
            outputs, _ = model_mylstm(inputs) 
            outputs = outputs[-1]  

            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Average Loss: {average_loss:.4f}')