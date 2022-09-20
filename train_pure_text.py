import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

# import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

from handle_data import get_text_data
from nets import PureTextModel
from MyDatasets import TextDataset


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name())

    batch_size = 64

    train_df, val_df, test_df = get_text_data.get_data('medium')
    print(train_df.columns)

    train_set = TextDataset.TextDataset(df=train_df)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)
    val_set = TextDataset.TextDataset(df=val_df)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4)

    train_num = len(train_set)
    val_num = len(val_set)
    print("Using {0} pairs for training, {1} pairs for validation.".format(train_num, val_num))

    save_path = 'saved_models/save_text_model.pth'
    best_path = 'saved_models/best_text_model.pth'

    model = PureTextModel.PureTextModel(num_classes=2,
                                        text_dim=512,
                                        claim_dim=512,
                                        fusion_output_size=1024)
    model.to(device)  # for model, to() function is an inplace function

    criterion = nn.BCEWithLogitsLoss()
    lr = 1e-5
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

    train_steps = len(train_loader)
    valid_steps = len(val_loader)
    train_losses = []
    total_train_steps = 0
    total_valid_steps = 0

    best_f1 = 0.0
    epochs = 50

    train_pred = []
    train_labels = []
    val_pred = []
    val_labels = []

    for epoch in range(epochs):
        train_bar = tqdm(train_loader)
        model.train()
        train_loss = 0

        for step, data in enumerate(train_bar):
            total_train_steps += 1

            claim_data = data['claim']
            text_data = data['text']
            labels = data['label']

            # for variables, to() function is not an inplace function
            claim_data = claim_data.to(device)
            text_data = text_data.to(device)
            labels = labels.to(device)
            # print(text_data.shape, image_data.shape)

            output = model(text_data, claim_data).squeeze()
            # print(output)
            # print(output.shape)

            # get predictions to calculate scores
            train_pred += [1 if torch.argmax(item) == 1 else 0 for item in output]
            train_labels += [1 if torch.argmax(item) == 1 else 0 for item in labels]

            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_losses.append(loss.item())

            train_bar.desc = "train epoch[{}/{}]  loss: {:.6f}  F1: {:.6f}".format(epoch + 1, epochs,
                                                                                   train_loss / (step + 1),
                                                                                   f1_score(train_labels, train_pred, average='macro'))

        # valid
        model.eval()
        valid_loss = 0.0
        valid_losses = []
        i = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader)

            for data in val_bar:
                i += 1

                claim_data = data['claim']
                text_data = data['text']
                labels = data['label']

                # for variables, to() function is not an inplace function
                claim_data = claim_data.to(device)
                text_data = text_data.to(device)
                labels = labels.to(device)
                # print(text_data.shape, image_data.shape)

                output = model(text_data, claim_data).squeeze()
                val_pred += [1 if item[1] == 1 else 0 for item in output]
                val_labels += [1 if item[1] == 1 else 0 for item in labels]

                loss = criterion(output, labels)
                valid_loss += loss.item()
                valid_losses.append(loss.item())

                val_bar.desc = "valid epoch[{}/{}]  loss: {:.6f}  F1: {:.6f}".format(epoch + 1, epochs,
                                                                                     valid_loss / i,
                                                                                     f1_score(val_labels, val_pred, average='macro'))

        # print('[epoch %d] train_loss: %.6f  valid_loss: %.6f' %
        #       (epoch + 1, train_loss / train_steps, valid_loss / valid_steps))

        scheduler.step()
        torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    main()
