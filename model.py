import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class ConvUIL(nn.Module):
    input_size = [340, 512]
    output_size = 13
    input_channels = 1
    channels_conv1 = 9
    channels_conv2 = 18
    kernel_conv1 = [3, 3]
    kernel_conv2 = [3, 3]
    pool_conv1 = [2, 2]
    pool_conv2 = [1, 2]
    fcl1_size = 50

    def __init__(self):
        super(ConvUIL, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(self.input_channels, self.channels_conv1, self.kernel_conv1)
        self.conv2 = nn.Conv2d(self.channels_conv1, self.channels_conv2, self.kernel_conv2)

        # Calculate the convolutional layers output size (stride = 1)
        c1 = np.array(self.input_size) - self.kernel_conv1 + 1
        p1 = c1 // self.pool_conv1
        c2 = p1 - self.kernel_conv2 + 1
        p2 = c2 // self.pool_conv2
        self.conv_out_size = int(p2[0] * p2[1] * self.channels_conv2)

        # Define the fully connected layers
        self.fcl1 = nn.Linear(self.conv_out_size, self.fcl1_size)
        self.fcl2 = nn.Linear(self.fcl1_size, self.output_size)

    def forward(self, x):
        # Apply convolution 1 and pooling
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, self.pool_conv1)

        # Apply convolution 2 and pooling
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, self.pool_conv2)

        # Reshape x to one dimmension to use as input for the fully connected layers
        x = x.view(-1, self.conv_out_size)

        # Fully connected layers
        x = self.fcl1(x)
        x = F.relu(x)
        x = self.fcl2(x)

        return x

if __name__ == '__main__':
    model = ConvUIL()
    print(model)