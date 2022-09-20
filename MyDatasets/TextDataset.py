import torch
from torch.utils.data import Dataset
from PIL import Image


class TextDataset(Dataset):

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df['text_emb'].iloc[idx]
        claim = self.df['claim_emb'].iloc[idx]
        num_replies = self.df['num_replies'].iloc[idx]
        num_retweets = self.df['num_retweets'].iloc[idx]
        lang = self.df['lang_emb'].iloc[idx]
        label = self.df['label'].iloc[idx]

        text = torch.Tensor(text).float()
        text = text.reshape(-1, 1)
        num_replies = torch.Tensor([num_replies]).float()
        num_replies = num_replies.reshape(-1, 1)
        # print(num_replies)
        num_retweets = torch.Tensor([num_retweets]).float()
        num_retweets = num_retweets.reshape(-1, 1)
        claim = torch.Tensor(claim).float()
        claim = claim.reshape(-1, 1)
        lang = torch.Tensor(lang).float()
        lang = lang.reshape(-1, 1)
        if label == 'misinformation':
            label = torch.Tensor([1., 0.])
        else:
            label = torch.Tensor([0., 1.])

        sample = {
            'text': text,
            'claim': claim,
            'num_replies': num_replies,
            'num_retweets': num_retweets,
            'lang': lang,
            'label': label
        }

        return sample
