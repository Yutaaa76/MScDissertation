import torch
import torch.nn as nn
# from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from adabelief_pytorch import AdaBelief

# import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

from handle_data import get_text_data
from nets import PureTextConvModel
from MyDatasets import TextDataset


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name())

    batch_size = 128

    # train_df, val_df, test_df = get_text_data.get_data(size='small')
    train_df, val_df, test_df = get_text_data.get_data(size='medium')

    train_set = TextDataset.TextDataset(df=train_df)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    val_set = TextDataset.TextDataset(df=val_df)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)
    test_set = TextDataset.TextDataset(df=test_df)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)

    train_num = len(train_set)
    val_num = len(val_set)
    test_num = len(test_set)
    print("Using {0} samples for training, {1} samples for validation, {2} samples for test.".format(train_num, val_num, test_num))

    save_path = 'saved_models/save_text_conv_model.pth'
    best_path = 'saved_models/best_text_conv_model.pth'

    model = PureTextConvModel.PureTextConvModel(num_classes=2,
                                                text_dim=768,
                                                claim_dim=768,
                                                lang_dim=46,  # small:41, medium:46
                                                conv_num=1024,
                                                fc_num=2048)
    model.to(device)  # for model, the to() function is an inplace function

    # weights = torch.Tensor([0.1, 0.9]).to(device)
    # criterion = nn.BCEWithLogitsLoss(weight=weights)
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()

    lr = 5e-5
    optimizer = AdaBelief(model.parameters(), lr=lr, print_change_log=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

    train_steps = len(train_loader)
    valid_steps = len(val_loader)
    train_losses = []
    total_train_steps = 0
    total_valid_steps = 0

    best_val_f1 = 0.0
    best_val_fac_f1 = 0.0
    best_test_f1 = 0.0
    best_test_fac_f1 = 0.0
    epochs = 25

    train_mis_f1 = []
    train_fac_f1 = []
    val_mis_f1 = []
    val_fac_f1 = []

    for epoch in range(epochs):
        train_bar = tqdm(train_loader, colour='cyan')
        model.train()
        train_loss = 0
        train_pred = []
        train_labels = []

        for step, data in enumerate(train_bar):
            total_train_steps += 1

            claim_data = data['claim']
            num_replies = data['num_replies']
            num_retweets = data['num_retweets']
            text_data = data['text']
            lang_data = data['lang']
            labels = data['label']

            # for variables, to() function is not an inplace function
            claim_data = claim_data.to(device)
            num_replies = num_replies.to(device)
            num_retweets = num_retweets.to(device)
            text_data = text_data.to(device)
            lang_data = lang_data.to(device)
            labels = labels.to(device)

            output = model(text_data, claim_data, lang_data, num_replies, num_retweets).squeeze()

            # get predictions to calculate scores
            # print(output[0])
            train_pred += [1 if torch.argmax(item) == 1 else 0 for item in output]
            train_labels += [1 if torch.argmax(item) == 1 else 0 for item in labels]
            # train_pred += [item[1] for item in output]
            # train_labels += [1 if torch.argmax(item) == 1 else 0 for item in labels]

            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_losses.append(loss.item())

            train_bar.desc = "train epoch[{}/{}]  loss: {:.6f}  Misinformation F1: {:.6f}  Factual F1:{:.6f}  Macro F1:{:.6f}".format(
                epoch + 1, epochs,
                train_loss / (step + 1),
                f1_score(train_labels, train_pred, average=None)[0],
                f1_score(train_labels, train_pred, average=None)[1],
                f1_score(train_labels, train_pred, average='macro'))  # f1_score(train_labels, train_pred, average='macro')

            train_mis_f1 += f1_score(train_labels, train_pred, average=None)[0]
            train_fac_f1 += f1_score(train_labels, train_pred, average=None)[1]

        # valid
        model.eval()
        valid_loss = 0.0
        valid_losses = []
        val_pred = []
        val_pred_mis = []
        val_pred_fac = []
        val_labels = []
        i = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, colour='magenta')

            for data in val_bar:
                i += 1

                claim_data = data['claim']
                num_replies = data['num_replies']
                num_retweets = data['num_retweets']
                text_data = data['text']
                lang_data = data['lang']
                labels = data['label']

                # for variables, to() function is not an inplace function
                claim_data = claim_data.to(device)
                num_replies = num_replies.to(device)
                num_retweets = num_retweets.to(device)
                text_data = text_data.to(device)
                lang_data = lang_data.to(device)
                labels = labels.to(device)
                # print(text_data.shape, image_data.shape)

                output = model(text_data, claim_data, lang_data, num_replies, num_retweets).squeeze()
                val_pred += [1 if torch.argmax(item) == 1 else 0 for item in output]
                val_labels += [1 if torch.argmax(item) == 1 else 0 for item in labels]

                loss = criterion(output, labels)
                valid_loss += loss.item()
                valid_losses.append(loss.item())

                val_bar.desc = "valid epoch[{}/{}]  loss: {:.6f}  Misinformation F1: {:.6f}  Factual F1:{:.6f}  Macro F1:{:.6f}".format(
                    epoch + 1, epochs,
                    valid_loss / i,
                    f1_score(val_labels, val_pred, average=None, labels=[0, 1])[0],
                    f1_score(val_labels, val_pred, average=None, labels=[0, 1])[1],
                    f1_score(val_labels, val_pred, average='macro'))  # f1_score(val_labels, val_pred, average='macro')

                val_mis_f1 += f1_score(val_labels, val_pred, average=None, labels=[0, 1])[0]
                val_fac_f1 += f1_score(val_labels, val_pred, average=None, labels=[0, 1])[1]

            if f1_score(val_labels, val_pred, average='macro') > best_val_f1:
                best_val_f1 = f1_score(val_labels, val_pred, average='macro')
                best_val_fac_f1 = f1_score(val_labels, val_pred, average=None, labels=[0, 1])[1]

        # test
        test_loss = 0.0
        test_losses = []
        test_pred = []
        test_pred_mis = []
        test_pred_fac = []
        test_labels = []
        i = 0
        with torch.no_grad():
            test_bar = tqdm(test_loader, colour='green')

            for data in test_bar:
                i += 1

                claim_data = data['claim']
                num_replies = data['num_replies']
                num_retweets = data['num_retweets']
                text_data = data['text']
                lang_data = data['lang']
                labels = data['label']

                # for variables, to() function is not an inplace function
                claim_data = claim_data.to(device)
                num_replies = num_replies.to(device)
                num_retweets = num_retweets.to(device)
                text_data = text_data.to(device)
                lang_data = lang_data.to(device)
                labels = labels.to(device)
                # print(text_data.shape, image_data.shape)

                output = model(text_data, claim_data, lang_data, num_replies, num_retweets).squeeze()
                test_pred += [1 if torch.argmax(item) == 1 else 0 for item in output]
                test_labels += [1 if torch.argmax(item) == 1 else 0 for item in labels]

                loss = criterion(output, labels)
                test_loss += loss.item()
                test_losses.append(loss.item())

                test_bar.desc = "test  epoch[{}/{}]  loss: {:.6f}  Misinformation F1: {:.6f}  Factual F1:{:.6f}  Macro F1:{:.6f}".format(
                    epoch + 1, epochs,
                    test_loss / i,
                    f1_score(test_labels, test_pred, average=None, labels=[0, 1])[0],
                    f1_score(test_labels, test_pred, average=None, labels=[0, 1])[1],
                    f1_score(test_labels, test_pred, average='macro'))  # f1_score(val_labels, val_pred, average='macro')

            if f1_score(test_labels, test_pred, average='macro') > best_test_f1:
                best_test_f1 = f1_score(test_labels, test_pred, average='macro')
                best_test_fac_f1 = f1_score(test_labels, test_pred, average=None, labels=[0, 1])[1]

        scheduler.step()
        torch.save(model.state_dict(), save_path)
        torch.cuda.empty_cache()
        print("best val f1: {:.6f}    best val fac f1: {:.6f}   best test f1: {:.6f}   best test fac f1:{:.6f}".format(
            best_val_f1,
            best_val_fac_f1,
            best_test_f1,
            best_test_fac_f1
        ))

    print(train_mis_f1)
    print(train_fac_f1)
    print(val_mis_f1)
    print(val_fac_f1)


if __name__ == '__main__':
    main()
