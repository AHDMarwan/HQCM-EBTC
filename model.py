import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

# ---------------------------
# Hybrid Quantum-Classical Model
# ---------------------------
n_qubits = 5
dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev, interface="torch")
def qnode(inputs, weights):
    qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True)
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits), imprimitive=qml.ops.CZ)
    return [qml.expval(qml.PauliY(wires=i)) for i in range(n_qubits)]
        
class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        # Channel Attention
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Pool to (batch, channels, 1, 1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // 4, in_channels, kernel_size=1)
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.conv_spatial = nn.Conv2d(in_channels, 1, kernel_size=1, padding=0)  # Output: (batch, 1, H, W)
        self.sigmoid_spatial = nn.Sigmoid()
        self.weights = nn.Parameter(torch.ones(3))  # Learnable weights for fusion

    def forward(self, x):
        # Channel Attention
        channel_avg = self.global_avg_pool(x)  # Shape: (batch, channels, 1, 1)
        channel_weights = self.fc1(channel_avg)
        channel_weights = self.fc2(channel_weights)
        channel_weights = self.sigmoid_channel(channel_weights)
        x = x * channel_weights  # Apply channel attention

        # Multi-scale Spatial Attention
        features1 = self.conv1(x)
        features2 = self.conv2(x)
        features3 = self.conv3(x)
        combined_features = (self.weights[0] * features1 + self.weights[1] * features2 + self.weights[2] * features3) / (self.weights.sum() + 1e-6)
        spatial_attention = self.conv_spatial(combined_features)
        spatial_attention = self.sigmoid_spatial(spatial_attention)
        spatial_attention = spatial_attention.expand_as(x)
        return spatial_attention  # Same shape as input: (batch, channels, H, W)

class HQNetPL(nn.Module):
    def __init__(self, n_qubits, circuit_depth, n_parallel_circuits, class_number):
        super(HQNetPL, self).__init__()
        # Convolutional feature extractor
        self.bn = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(32 * 16 * 16, 2 ** n_qubits * n_parallel_circuits)
        weight_shapes = {"weights": (circuit_depth, n_qubits, 3)}
        self.qlayers = nn.ModuleList([qml.qnn.TorchLayer(qnode, weight_shapes) for _ in range(n_parallel_circuits)])
        self.fc3 = nn.Linear(n_qubits * n_parallel_circuits, class_number)
        self.attention = AttentionModule(in_channels=32)  # Attention is applied to the 32 channels from conv3
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) #
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm2d): #added batchnorm initialization
                nn.init.constant_(m.weight, 1) #gamma(weight) to 1
                nn.init.constant_(m.bias, 0) #beta (bias) to 0
    def feature_extractor(self, x):
        """ Extract features before attention. """
        x = self.bn(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))  # Output after conv3
        x = self.pool(x)
        x = self.dropout(x)
        return x  # Returns feature maps before attention
    def forward(self, x):
        x = self.feature_extractor(x)
        attention_map = self.attention(x)
        x = x * (1 + attention_map)  # Normalize attention map to [0, 1]
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        n_parallel = len(self.qlayers)
        x_split = torch.chunk(x, n_parallel, dim=1)
        quantum_outputs = torch.stack([layer(x_i) for layer, x_i in zip(self.qlayers, x_split)], dim=1)
        x = quantum_outputs.view(x.size(0), -1)
        x = self.fc3(x)
        return x

