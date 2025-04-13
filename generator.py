import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, condition_dim, noise_dim, img_channels=3, feature_map_size=64):
        """
        Conditional Generator
        Args:
            condition_dim (int): Dimension of the text condition (e.g., 512 from CLIP).
            noise_dim (int): Dimension of the random noise vector (e.g., 100).
            img_channels (int): Number of channels in the output image (3 for RGB).
            feature_map_size (int): Base number of features for each layer.
        """
        super(Generator, self).__init__()
        self.condition_dim = condition_dim
        self.noise_dim = noise_dim
        input_dim = noise_dim + condition_dim  # Concatenation of noise and text condition.

        self.net = nn.Sequential(
            # Input: (input_dim) x 1 x 1 -> Output: (feature_map_size*8) x 4 x 4
            nn.ConvTranspose2d(input_dim, feature_map_size * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.ReLU(True),
            # Output: (feature_map_size*8) x 4 x 4 -> (feature_map_size*4) x 8 x 8
            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),
            # Output: (feature_map_size*4) x 8 x 8 -> (feature_map_size*2) x 16 x 16
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),
            # Output: (feature_map_size*2) x 16 x 16 -> (feature_map_size) x 32 x 32
            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),
            # Output: (feature_map_size) x 32 x 32 -> (img_channels) x 64 x 64
            nn.ConvTranspose2d(feature_map_size, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Final activation to bring outputs to range [-1, 1]
        )
    
    def forward(self, noise, condition):
        """
        Forward pass for the generator.
        Args:
            noise (Tensor): Random noise of shape [batch_size, noise_dim].
            condition (Tensor): Text condition vector of shape [batch_size, condition_dim].
        Returns:
            Tensor: Generated image of shape [batch_size, img_channels, 64, 64].
        """
        # Concatenate noise and condition vectors along the feature dimension.
        combined = torch.cat((noise, condition), dim=1)
        # Reshape to (batch_size, input_dim, 1, 1) for convolutional layers.
        combined = combined.unsqueeze(2).unsqueeze(3)
        # Generate image through the network.
        output = self.net(combined)
        return output

# For quick testing of the generator.
if __name__ == "__main__":
    noise = torch.randn(1, 100)
    condition = torch.randn(1, 512)
    model = Generator(condition_dim=512, noise_dim=100)
    fake_image = model(noise, condition)
    print(fake_image.shape)  # Expected: torch.Size([1, 3, 64, 64])
