import torch
import torch.nn as nn


class SimpleTCN(nn.Module):
    """En enkel TCN-lignende modell for point forecasts."""

    def __init__(
        self,
        input_channels=1,
        hidden_channels=16,
        kernel_size=3,
        input_length=28,
        output_size=7,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.fc = nn.Linear(hidden_channels * input_length, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)  # flatten
        output = self.fc(x)

        return output


class QuantileTCN(nn.Module):
    """TCN som predikerer flere kvantiler for hver dag i forecast-horisonten."""

    def __init__(
        self,
        input_channels=1,
        hidden_channels=16,
        kernel_size=3,
        input_length=28,
        output_size=7,
        num_quantiles=3,
    ):
        super().__init__()

        padding = kernel_size // 2
        self.output_size = output_size
        self.num_quantiles = num_quantiles

        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.fc = nn.Linear(hidden_channels * input_length, output_size * num_quantiles)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)
        output = self.fc(x)
        output = output.view(-1, self.num_quantiles, self.output_size)

        # Sikrer monotone kvantiler: q0.1 <= q0.5 <= q0.9
        output, _ = torch.sort(output, dim=1)
        return output


if __name__ == "__main__":
    model = SimpleTCN()

    dummy_input = torch.randn(12, 1, 28)  # batch_size=12, input_channels=1, sequence_length=28
    output = model(dummy_input)

    print("Input shape:", dummy_input.shape)  # should be (12, 1, 28)
    print("Output shape:", output.shape)  # should be (12, 7) for output_size=7
