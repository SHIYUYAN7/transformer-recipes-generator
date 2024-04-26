import torch
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from LSTM import ComplexLSTM as LSTM
    

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)


def generate_recipe(model, tokenizer, ingredients, max_length=128):
    model.eval()  # eval mode
    ingredients = ' '.join(ingredients)  # change ingredient to string
    input_ids = tokenizer.encode(ingredients, return_tensors='pt')

    generated_ids = torch.full((input_ids.size(0), 1), tokenizer.cls_token_id, dtype=torch.long)
    with torch.no_grad():
        for _ in range(max_length):
            # mask
            current_len = generated_ids.size(1)
            tgt_mask = model.generate_square_subsequent_mask(current_len)

            # predict, get last output
            output = model(input_ids, generated_ids)
            next_token_logits = output[:, -1, :]
            # print(next_token_logits)
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

            # if its end tag
            if next_token_id.item() == tokenizer.sep_token_id:
                break

    # back to text
    generated_text = tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True)
    print('generated_ids:', generated_ids.squeeze())
    return generated_text


if __name__ == "__main__":
    data = pd.read_excel("../final_project/recipes_20000.xlsx")
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

    model = LSTM(vocab_size, 128, 256, 2)
    # TensorDataset
    dataset = TensorDataset(input_ids_tensor, output_ids_tensor)

    # create DataLoader
    batch_size = 64
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()

    epochs = 5
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs, targets

            # Forward pass
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Get model outputs for the inputs
            # targets[:, 1:] to align targets to the predictions (shifting for prediction of next token)
            loss = criterion(outputs.transpose(1, 2), targets[:, 1:])  # Calculate loss

            # Backward pass
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the model's weights

            total_loss += loss.item()  # Accumulate the loss

        # After finishing all batches in the DataLoader
        average_loss = total_loss / len(train_loader)  # Calculate the average loss
        print(f'Epoch {epoch + 1}, Average Loss: {average_loss:.4f}')

    # test
    ingredients_list = ['onion', 'red bell pepper', 'garlic cloves']

    # generate
    recipe_text = generate_recipe(model, tokenizer, ingredients_list)
    print(recipe_text)




