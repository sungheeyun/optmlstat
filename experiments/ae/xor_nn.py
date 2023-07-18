
from matplotlib import pyplot as plt
import torch
from torch import optim
import torch.nn as nn
from torch.nn import Module, Linear
import torchvision


class XOR(Module):

    HIDDEN_DIM: int = 128
    EMBEDDING_DIM: int = 128

    def __init__(self) -> None:
        super().__init__()

        self.hidden_layer: Module = Linear(in_features=2, out_features=2)
        self.output_layer: Module = Linear(in_features=2, out_features=1)

    def forward(self, features):

        y_values = self.hidden_layer(features)
        z_values = torch.relu(y_values)
        final_values = self.output_layer(z_values)

        return final_values


if __name__ == "__main__":

    input_shape: int = 28 ** 2
    train_batch_size: int = 128
    test_batch_size: int = 32
    num_workers: int = 4
    num_total_epochs: int = 20

    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a model from "AE" autoencoder class
    # load it to the specified device, either gpu or cpu
    model = XOR().to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e=3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # mean-squared error loss
    criterion = nn.MSELoss()

    # get data set

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )

    train_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=True, transform=transform, download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets",
        train=False,
        transform=transform,
        download=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # training

    for epoch in range(num_total_epochs):
        loss: float = 0.0
        inner_cnt: int = 0
        for batch_features_, _ in train_loader:
            inner_cnt += 1
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            batch_features = batch_features_.view(-1, input_shape).to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs = model(batch_features)

            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(train_loader)

        # display the epoch training loss
        print(f"epoch: {epoch + 1}/{num_total_epochs}, loss = {loss:.6f}")

    for features_, _ in test_loader:
        features = features_.view(-1, input_shape)
        break

    dec_features = model(features)

    feat_array = features.numpy()
    dec_feat_array = dec_features.detach().numpy()

    fig, ax_array = plt.subplots(2, 5)

    for idx in range(ax_array.shape[1]):
        ax_array[0, idx].imshow(feat_array[idx, :].reshape(28, 28))
        ax_array[1, idx].imshow(dec_feat_array[idx, :].reshape(28, 28))

    fig.show()
