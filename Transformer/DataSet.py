from torch.utils.data import Dataset
import pandas as pd


class RecipeDataset(Dataset):
    def __init__(self, file_path, tokenizer, head=20000):
        self.data = pd.read_excel(file_path).head(head)
        self.tokenizer = tokenizer
        self.ingredients, self.steps = [], []
        for ingredient, step in zip(self.data['ingredients'], self.data['steps']):
            ingredient_tokens = self.tokenizer(' '.join(eval(ingredient)), padding="max_length", truncation=True, max_length=64, return_tensors="pt")
            step_tokens = self.tokenizer(preprocess_text(' '.join(eval(step))), padding="max_length", truncation=True, max_length=256, return_tensors="pt")
            self.ingredients.append({key: val.squeeze(0) for key, val in ingredient_tokens.items()})
            self.steps.append({key: val.squeeze(0) for key, val in step_tokens.items()})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.ingredients[idx], self.steps[idx]


def preprocess_text(text):
    return text.replace("&amp;", "")