import torch
import torch.nn.functional as F


import umap
# from tqdm import tqdm
# from sklearn.metrics import f1_score

from handle_data import get_dgl_data
from nets import HGCN


def main():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(torch.cuda.get_device_name())

    data = get_dgl_data.get_data(size='small')
    data = data.to(device)
    # print(data.nodes)
    # print(data.canonical_etypes)
    model = HGCN.RGCN(774, 512, 2, data.etypes).to(device)

    lr = 3e-4

    reducer = umap.UMAP(n_components=774)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    user_feats = data.nodes['user'].data['feat'].float()
    # hashtag_feats = torch.ones(28244, 774).cuda()
    # print(1)
    # reply_feats = torch.from_numpy(reducer.fit_transform(data.nodes['reply'].data['feat'].float().cpu().detach().numpy())).cuda()
    print(1)
    claim_feats = torch.from_numpy(reducer.fit_transform(data.nodes['claim'].data['feat'].float().cpu().detach().numpy())).to(device)
    # print(1)
    # article_feats = torch.from_numpy(reducer.fit_transform(data.nodes['article'].data['feat'].float().cpu().detach().numpy())).cuda()
    # print(1)
    # image_feats = torch.from_numpy(reducer.fit_transform(data.nodes['image'].data['feat'].float().cpu().detach().numpy())).cuda()
    print(1)
    tweet_feats = torch.from_numpy(reducer.fit_transform(data.nodes['tweet'].data['feat'].float().cpu().detach().numpy())).to(device)
    print(1)

    labels = data.nodes['tweet'].data['label']
    train_mask = data.nodes['tweet'].data['train_mask']
    # val_mask = data.nodes['tweet'].data['val_mask']

    node_features = {
        'user': user_feats,
        'tweet': tweet_feats,
        'claim': claim_feats,
    }

    for epoch in range(10):
        model.train()
        logits = model(data, node_features)['tweet']
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())


if __name__ == '__main__':
    main()
