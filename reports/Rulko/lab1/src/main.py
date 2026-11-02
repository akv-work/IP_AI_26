from data import get_data_loaders
from model import CNN
from train import train_model
from evaluate import evaluate_model
from visualize import plot_loss, show_prediction


def main():
    train_loader, test_loader, train_dataset, test_dataset = get_data_loaders(batch_size=64)

    model = CNN()

    epoch_losses = train_model(model, train_loader, num_epochs=5, lr=1.0)

    evaluate_model(model, test_loader)

    plot_loss(epoch_losses)
    show_prediction(model, test_dataset)


if __name__ == "__main__":
    main()
