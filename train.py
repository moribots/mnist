"""
File: train.py
Description: Main script to train and evaluate the MLP model on MNIST.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse  # For parsing command-line arguments
import wandb  # For experiment tracking with Weights & Biases

# Import our custom modules
from model import MLP
from dataset_utils import get_mnist_dataloaders


def train(model, device, train_loader, optimizer, criterion, epoch, log_interval):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        device (torch.device): The device to train on (e.g., 'cuda' or 'cpu').
        train_loader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (nn.Module): The loss function.
        epoch (int): The current epoch number.
        log_interval (int): How often to log training progress.
    """
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # Move to device.
        data, target = data.to(device), target.to(device)

        # Zero the gradients.
        # They are accumulated by default, so we need to clear them before each backprop.
        optimizer.zero_grad()

        # Forward pass.
        # Get the model's predictions (logits) for the input data.
        output = model(data)

        # Compute the loss.
        # Compare the  model's predictions with the true labels (target).
        loss = criterion(output, target)
        # Multiply by batch size to get total loss for the batch
        running_loss += loss.item() * data.size(0)

        # Backward pass.
        # Compute the loss gradients wrt the model parameters.
        loss.backward()

        # Optimizer step.
        # Update the model parameters based on the computed gradients.
        optimizer.step()

        # Compute the number of correct predictions to collect metrics.
        # Get the index of the max log-probability.
        pred = output.argmax(dim=1, keepdim=True)
        # Compare the predicted labels with the true labels.
        correct_predictions += pred.eq(target.view_as(pred)).sum().item()
        total_samples += data.size(0)

        # Log to W&B.
        if batch_idx % log_interval == 0:
            current_loss = loss.item()
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {current_loss:.6f}")
            # Log batch loss to W&B
            wandb.log({
                "epoch": epoch,
                "batch_train_loss": current_loss,
                "batch_idx": batch_idx
            })

    avg_epoch_loss = running_loss / total_samples
    epoch_accuracy = 100. * correct_predictions / total_samples
    return avg_epoch_loss, epoch_accuracy


def evaluate(model, device, test_loader, criterion, epoch):
    """
    Evaluates the model on the test dataset.

    Args:
        model (nn.Module): The neural network model.
        device (torch.device): The device to evaluate on.
        test_loader (DataLoader): DataLoader for the test data.
        criterion (nn.Module): The loss function.
        epoch (int): Current epoch (for logging purposes).

    Returns:
        tuple: Average test loss and test accuracy.
    """
    # Set the model to evaluation mode.
    model.eval()
    test_loss = 0.0
    correct_predictions = 0
    with torch.no_grad():  # Disable gradient computations for efficiency.
        for data, target in test_loader:
            # Move to device.
            data, target = data.to(device), target.to(device)
            # Forward pass.
            output = model(data)
            test_loss += criterion(output, target).item() * \
                data.size(0)  # Multiply by batch size
            # Compute the number of correct predictions. Get the index of the max log-probability.
            pred = output.argmax(dim=1, keepdim=True)
            correct_predictions += pred.eq(target.view_as(pred)).sum().item()

    # Average loss over the entire test set
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct_predictions / \
        len(test_loader.dataset)  # Accuracy as a percentage

    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct_predictions}/{len(test_loader.dataset)} "
          f"({test_accuracy:.2f}%)\n")

    # Log test metrics to W&B
    wandb.log({
        "epoch": epoch,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    })
    return test_loss, test_accuracy


def main():
    # --------------------------------------------------------------------------
    # 1. Argument Parsing
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="PyTorch MLP MNIST Example")
    parser.add_argument('--input-size', type=int, default=784, metavar='N',
                        help='input size for MLP (default: 28*28=784)')
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[128, 64],
                        help='list of hidden layer sizes (default: [128, 64])')
    parser.add_argument('--output-size', type=int, default=10, metavar='N',
                        help='output size for MLP (number of classes, default: 10 for MNIST)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status (default: 10)')
    parser.add_argument('--wandb-project', type=str, default="mlp-mnist-example",
                        help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default=None,  # Replace with your W&B entity if needed
                        help='Weights & Biases entity (username or team)')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory to store MNIST data')
    parser.add_argument('--save-model-path', type=str, default='./projects/phase1_week1_mlp_mnist/saved_models/mnist_mlp.pth',
                        help='Path to save the trained model')

    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # 2. Setup (Device, Seed, W&B)
    # --------------------------------------------------------------------------
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)  # Seed for all GPUs if used

    # Initialize Weights & Biases
    wandb.init(project=args.wandb_project,
               entity=args.wandb_entity, config=args)
    wandb.watch_called = False  # To avoid issues with re-watching

    # --------------------------------------------------------------------------
    # 3. Load Data
    # --------------------------------------------------------------------------
    train_loader, test_loader = get_mnist_dataloaders(
        args.batch_size, data_dir=args.data_dir)

    # --------------------------------------------------------------------------
    # 4. Initialize Model, Loss, Optimizer
    # --------------------------------------------------------------------------
    model = MLP(input_size=args.input_size,
                hidden_sizes=args.hidden_sizes,
                output_size=args.output_size).to(device)

    # Log model architecture and gradients/parameters to W&B
    # Note: `wandb.watch()` should be called after model is on device and before training loop
    wandb.watch(model, log="all", log_freq=args.log_interval)

    criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # You could also try SGD: optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # --------------------------------------------------------------------------
    # 5. Training and Evaluation Loop
    # --------------------------------------------------------------------------
    best_test_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(
            model, device, train_loader, optimizer, criterion, epoch, args.log_interval)
        test_loss, test_acc = evaluate(
            model, device, test_loader, criterion, epoch)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # Log epoch-level metrics to W&B
        wandb.log({
            "epoch": epoch,
            "epoch_train_loss": train_loss,
            "epoch_train_accuracy": train_acc,
            # Test loss and accuracy are already logged in the evaluate function
        })

        # Save the model if it's the best one so far
        if test_acc > best_test_accuracy:
            best_test_accuracy = test_acc
            try:
                torch.save(model.state_dict(), args.save_model_path)
                print(
                    f"Best model saved to {args.save_model_path} (Accuracy: {best_test_accuracy:.2f}%)")
                # You can also save it as a W&B artifact
                # wandb.save(args.save_model_path) # This saves it to W&B run files
            except Exception as e:
                print(f"Error saving model: {e}")

    wandb.finish()  # Finish the W&B run
    print("Training complete.")
    print(f"Best Test Accuracy: {best_test_accuracy:.2f}%")
    print(f"Model saved at: {args.save_model_path}")


if __name__ == '__main__':
    main()
