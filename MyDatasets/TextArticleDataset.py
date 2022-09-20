import torch
from torch.utils.data import Dataset
from PIL import Image


class MultimodalDataset(Dataset):

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df['text_emb'].iloc[idx]
        claim = self.df['claim_emb'].iloc[idx]
        title = self.df['title_emb'].iloc[idx]
        content = self.df['content_emb'].iloc[idx]
        lang = self.df['lang_emb'].iloc[idx]
        label = self.df['label'].iloc[idx]

        # lstm input shape: (batch_size, sequence_length, input_size)
        text = torch.Tensor(text).float()
        text = text.reshape(-1, 1)
        claim = torch.Tensor(claim).float()
        claim = claim.reshape(-1, 1)
        title = torch.Tensor(title).float()
        title = title.reshape(-1, 1)
        content = torch.Tensor(content).float()
        content = content.reshape(-1, 1)
        lang = torch.Tensor(lang).float()
        lang = lang.reshape(-1, 1)
        if label == 'misinformation':
            label = torch.Tensor([1., 0.])
        else:
            label = torch.Tensor([0., 1.])

        sample = {
            'text': text,
            'claim': claim,
            'title': title,
            'content': content,
            'label': label,
            'lang': lang
        }

        return sample
