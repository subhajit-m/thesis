# references
# https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
# https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632
# https://stackoverflow.com/questions/53673575/in-place-shuffle-torch-tensor-in-the-order-of-a-numpy-ndarray


from dataset import Dataset
import torch
from model.VIT import MyViT
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
import numpy as np


np.random.seed(0)
torch.manual_seed(0)

downloadLocation = 'D:\\dke\\thesis\\environment\\datasets\\mnist'
image_dim = (1, 224, 224)


def trainer():
    data = Dataset(downloadLocation=downloadLocation, image_dim=image_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device,
          f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    # model = MyViT(image_dim, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(device)
    model = MyViT(image_dim, n_patches=16, n_blocks=2,
                  hidden_d=8, n_heads=2, out_d=10, mask_ratio=.75).to(device)

    N_EPOCHS = 5
    LR = 0.005

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(data.train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(data.train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

        # Test loop
        with torch.no_grad():
            correct, total = 0, 0
            test_loss = 0.0
            for batch in tqdm(data.test_loader, desc="Testing"):
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                test_loss += loss.detach().cpu().item() / len(data.test_loader)

                correct += torch.sum(torch.argmax(y_hat, dim=1)
                                     == y).detach().cpu().item()
                total += len(x)
            print(f"Test loss: {test_loss:.2f}")
            print(f"Test accuracy: {correct / total * 100:.2f}%")


trainer()
