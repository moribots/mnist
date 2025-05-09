"""
File: model.py
Description: Defines the MLP model architecture for MNIST classification.
"""

import torch
import torch.nn as nn
# Contains activation functions like ReLU, and also functional forms of layers
import torch.nn.functional as F
import einops


class MLP(nn.Module):
    """Multi-Layer Perceptron (MLP) for MNIST classification.

    Args:
        nn (Module): PyTorch neural network module.
    """

    def __init__(self, input_size, hidden_sizes, output_size):
        """
            Initializes the MLP model layers.

            Args:
                input_size (int): The dimensionality of the input features.
                                  For MNIST, this is 28*28 = 784 (flattened image).
                hidden_sizes (list of int): A list containing the number of neurons
                                            in each hidden layer.
                output_size (int): The number of output classes. For MNIST, this is 10 (digits 0-9).
        """
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # --------------------------------------------------------------------------
        # 1. Define Layers
        # --------------------------------------------------------------------------
        # We'll create a list of layers and then use nn.Sequential to wrap them.
        # `nn.Linear(in_features, out_features)` applies a linear transformation.
        # Activation functions (like ReLU) are typically applied after linear layers.
        # --------------------------------------------------------------------------
        layers = []

        # First hidden layer. Note that the input is input_size, whereas subsequent layers have hidden_sizes[i - 1] as input.
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        # Add hidden layers
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # `nn.Sequential` is a container module that chains modules together.
        # The layers defined above will be executed in sequence.
        self.network = nn.Sequential(*layers)

        # Note on output activation:
        # For multi-class classification with CrossEntropyLoss, you typically don't
        # need a Softmax activation function at the end of the model because
        # `nn.CrossEntropyLoss` internally applies `log_softmax`.
        # If you were to use `nn.NLLLoss` (Negative Log Likelihood Loss),
        # you would need `nn.LogSoftmax()` as the final layer.

        print(
            f"MLP model initialized with input_size={input_size}, hidden_sizes={hidden_sizes}, output_size={output_size}")
        print("Model architecture:")
        print(self.network)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor. For MNIST, its shape is typically
                              [batch_size, 1, 28, 28] (batch, channels, height, width).

        Returns:
            torch.Tensor: The output tensor (logits). Shape: [batch_size, output_size].
        """
        # --------------------------------------------------------------------------
        # 1. Reshape/Flatten Input
        # --------------------------------------------------------------------------
        # The input `x` is likely a batch of images (e.g., [batch_size, 1, 28, 28]).
        # Linear layers expect input of shape [batch_size, num_features].
        # So, we need to flatten the image dimensions (1x28x28) into a single vector (784).

        if x.ndim == 4:  # Check if it's a batch of images
            # Assumes input is (batch, channels, height, width)
            x = einops.rearrange(x, 'b c h w -> b (c h w)')

        # Pass the flattened input through the network
        logits = self.network(x)
        return logits


if __name__ == '__main__':
    # Simple test for the MLP model
    print("\nTesting MLP model...")

    # Define some dummy parameters
    batch_size = 4096
    input_channels = 1  # MNIST is grayscale
    img_height = 28
    img_width = 28
    input_feature_size = input_channels * img_height * img_width  # 1 * 28 * 28 = 784

    # Two hidden layers with 128 and 64 neurons
    hidden_layer_config = [128, 64]
    num_classes = 10  # 10 digits for MNIST

    # Create a model instance
    model = MLP(input_size=input_feature_size,
                hidden_sizes=hidden_layer_config, output_size=num_classes)

    # Create a dummy input tensor (simulating a batch of MNIST images)
    # Shape: [batch_size, channels, height, width]
    dummy_input = torch.randn(
        batch_size, input_channels, img_height, img_width)
    print(f"\nDummy input tensor shape: {dummy_input.shape}")

    # Perform a forward pass
    output = model(dummy_input)
    # Expected: [batch_size, num_classes]
    print(f"Output tensor shape: {output.shape}")

    # Check if output shape is correct
    assert output.shape == (
        batch_size, num_classes), "Output shape is incorrect!"

    print("\n`model.py` test complete. Model seems to work as expected.")
