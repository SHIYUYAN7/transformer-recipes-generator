import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

from DataSet import RecipeDataset
import train, plot, generate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
optimizer = Adam(model.parameters(), lr=1e-4)

file_path = '/content/data.xlsx'
dataset = RecipeDataset(file_path, tokenizer, head=500)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(15):
    loss = train.train(model, dataloader, optimizer)
    perplexity = train.calculate_perplexity(model, dataloader, device)
    print(f"Epoch {epoch} Loss: {loss}")
    print(f"Epoch {epoch} Perplexity: {perplexity}")

plot.plot_loss_and_perplexity(losses=loss, perplexities=perplexity)

src_text = ' '.join(['eggs', 'all-purpose flour', 'white cornmeal', 'sugar', 'buttermilk', 'baking powder', 'baking soda', 'salt', 'canola oil', 'shoe peg corn'])

print('generated recipe: ', generate.generate_recipe(src_text, model, tokenizer))
