import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm
from sklearn.metrics import f1_score

from handle_data import get_text_img_data
from nets import TextAndVisionConcat
from MyDatasets import TextImageDataset


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 64

    # image data augmentation
    data_transform = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop([224, 224]),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_df, val_df, test_df = get_text_img_data.get_data(size='medium')
    print(train_df.columns)

    train_set = TextImageDataset.MultimodalDataset(df=train_df, transform=data_transform["train"])
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)
    val_set = TextImageDataset.MultimodalDataset(df=val_df, transform=data_transform["val"])
    val_loader = DataLoader(dataset=val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0)
    test_set = TextImageDataset.MultimodalDataset(df=test_df, transform=data_transform["test"])
    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0)

    train_num = len(train_set)
    val_num = len(val_set)
    print("Using {0} pairs for training, {1} pairs for validation.".format(train_num, val_num))

    save_path = 'saved_models/save_model.pth'
    best_path = 'saved_models/best_model.pth'

    valid_size = batch_size
    lr = 4e-5

    model = TextAndVisionConcat.TextAndVisionConcat(num_classes=2,
                                                    text_dim=768,
                                                    claim_dim=768,
                                                    vision_dim=512,
                                                    fusion_output_size=1024)
    model.to(device)  # for model, to() function is an inplace function
    # vis_model = model.vis_model
    for param in model.vis_model.parameters():
        param.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_steps = len(train_loader)
    valid_steps = len(val_loader)
    train_losses = []
    total_train_steps = 0
    total_valid_steps = 0

    best_acc = 0.0
    epochs = 50

    best_val_f1 = 0.0
    best_val_fac_f1 = 0.0
    best_test_f1 = 0.0
    best_test_fac_f1 = 0.0

    for epoch in range(epochs):
        train_bar = tqdm(train_loader, colour='cyan')
        model.train()
        train_loss = 0
        train_pred = []
        train_labels = []
        val_pred = []
        val_labels = []

        for step, data in enumerate(train_bar):
            total_train_steps += 1

            image_data = data['image']
            text_data = data['text']
            claim_data = data['claim']
            title_data = data['title']
            content_data = data['content']
            labels = data['label']

            # for variables, to() function is not an inplace function
            image_data = image_data.to(device)
            text_data = text_data.to(device)
            claim_data = claim_data.to(device)
            title_data = title_data.to(device)
            content_data = content_data.to(device)
            labels = labels.to(device)
            # print(text_data.shape, image_data.shape)

            output = model(image_data, text_data, claim_data, title_data, content_data).squeeze()

            train_pred += [1 if torch.argmax(item) == 1 else 0 for item in output]
            train_labels += [1 if torch.argmax(item) == 1 else 0 for item in labels]

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

                image_data = data['image']
                text_data = data['text']
                claim_data = data['claim']
                title_data = data['title']
                content_data = data['content']
                labels = data['label']

                # for variables, to() function is not an inplace function
                image_data = image_data.to(device)
                text_data = text_data.to(device)
                claim_data = claim_data.to(device)
                title_data = title_data.to(device)
                content_data = content_data.to(device)
                labels = labels.to(device).squeeze()
                # print(text_data.shape, image_data.shape)

                output = model(image_data, text_data, claim_data, title_data, content_data).squeeze()
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

                    image_data = data['image']
                    text_data = data['text']
                    claim_data = data['claim']
                    title_data = data['title']
                    content_data = data['content']
                    labels = data['label']

                    # for variables, to() function is not an inplace function
                    image_data = image_data.to(device)
                    text_data = text_data.to(device)
                    claim_data = claim_data.to(device)
                    title_data = title_data.to(device)
                    content_data = content_data.to(device)
                    labels = labels.to(device).squeeze()
                    # print(text_data.shape, image_data.shape)

                    output = model(image_data, text_data, claim_data, title_data, content_data).squeeze()
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

        # print('[epoch %d] train_loss: %.6f  valid_loss: %.6f' %
        #       (epoch + 1, train_loss / train_steps, valid_loss / valid_steps))

        torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    main()
