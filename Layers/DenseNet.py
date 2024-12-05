import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    """
    A single Dense Layer.
    """
    def __init__(self, input_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(input_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        return torch.cat([x, out], dim=1)  # Concatenate input with output

class DenseBlock(nn.Module):
    """
    A Dense Block consisting of multiple Dense Layers.
    """
    def __init__(self, num_layers, input_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(input_channels + i * growth_rate, growth_rate))
        self.dense_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.dense_block(x)

class TransitionLayer(nn.Module):
    """
    Transition Layer for downsampling between Dense Blocks.
    """
    def __init__(self, input_channels, output_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        return self.pool(x)

class DenseNet(nn.Module):
    """
    DenseNet implementation.
    """
    def __init__(self, growth_rate=32, num_init_features=64, block_config=(6, 12, 24, 16), num_classes=10):
        """
        Args:
            growth_rate (int): Number of filters added per layer.
            num_init_features (int): Number of filters in the initial convolution layer.
            block_config (tuple): Number of layers in each dense block.
            num_classes (int): Number of output classes.
        """
        super(DenseNet, self).__init__()

        # Initial Convolution
        self.features = nn.Sequential(
            nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Dense Blocks and Transition Layers
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            self.features.add_module(f'denseblock{i+1}', DenseBlock(num_layers, num_features, growth_rate))
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:  # Add transition layer except after the last dense block
                self.features.add_module(f'transition{i+1}', TransitionLayer(num_features, num_features // 2))
                num_features = num_features // 2

        # Final BatchNorm
        self.features.add_module('final_bn', nn.BatchNorm2d(num_features))

        # Classification Layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.adaptive_avg_pool2d(features, (1, 1))  # Global Average Pooling
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

# Example Usage
if __name__ == "__main__":
    model = DenseNet(growth_rate=32, num_init_features=64, block_config=(6, 12, 24, 16), num_classes=10)
    input_tensor = torch.randn(8, 1, 224, 224)  # Batch size of 8, single-channel images, size 224x224
    output = model(input_tensor)
    print("Output shape:", output.shape)  # Should be (8, 10)