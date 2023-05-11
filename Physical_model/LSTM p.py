import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define LSTM model for vehicle trajectory prediction
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, input):
        # Reshape input tensor to (batch_size, sequence_length, input_size)
        input = input.view(input.shape[0], input.shape[1], 2)

        # Set initial hidden state and cell state
        h0 = torch.zeros(1, input.size(0), self.lstm.hidden_size).to(device)
        c0 = torch.zeros(1, input.size(0), self.lstm.hidden_size).to(device)

        # Forward pass through LSTM layer
        output, (hn, cn) = self.lstm(input, (h0, c0))

        # Pass last output through linear layer to get predictions
        preds = self.fc(output[:, -1, :])

        return preds


# Generate vehicle trajectory data
def generate_data(num_cars=3, traj_length=3000):
    # Generate random starting positions for each car
    starting_pos = np.zeros((num_cars, 2))
    starting_pos[:, 0] = np.random.uniform(low=0, high=500000, size=num_cars)  # x_utm
    starting_pos[:, 1] = np.random.uniform(low=0, high=7000000, size=num_cars)  # y_utm

    # Generate random velocities for each car
    velocities = np.random.uniform(low=25, high=80, size=num_cars)  # km/h

    # Calculate time it would take for each car to travel 1km at its velocity
    times = 1 / (velocities / 60)  # minutes

    # Calculate total time it would take for each car to complete the trajectory
    total_time = (traj_length * 0.1) / 60  # 0.1 is the time step in seconds

    # Calculate the distance each car will travel during the trajectory
    distances = total_time * velocities

    # Calculate the angle each car will travel at
    angles = np.random.uniform(low=0, high=360, size=num_cars)

    # Convert angles to radians
    angles = np.radians(angles)

    # Calculate the change in x and y for each car for each time step
    delta_x = distances * np.cos(angles) / 1000  # km to m
    delta_y = distances * np.sin(angles) / 1000  # km to m

    # Initialize trajectory arrays
    x_traj = np.zeros((num_cars, traj_length))
    y_traj = np.zeros((num_cars, traj_length))

    # Generate trajectories
    for i in range(num_cars):
        x_traj[i, 0] = starting_pos[i, 0]
        y_traj[i, 0] = starting_pos[i, 1]

        for j in range(1, traj_length):
            x_traj[i, j] = x_traj[i, j - 1] + delta_x[i]
            y_traj[i, j] = y_traj[i, j - 1] + delta_y[i]

    # Split data into training and testing sets
    x_train, y_train, x_test, y_test = split_data(x_traj, y_traj, train_frac=0.8, shuffle=True)

    return x_train, y_train, x_test, y_test


# Split data into training and testing sets
def split_data(x_traj, y_traj, train_frac=0.8, shuffle=True):
    num_cars = x_traj.shape[0]
    traj_length = x_traj.shape[1]

    indices = np.arange(traj_length)

    if shuffle:
        np.random.shuffle(indices)

    train_indices = indices[:int(train_frac * traj_length)]
    test_indices = indices[int(train_frac * traj_length):]

    x_train = np.zeros((num_cars, len(train_indices) - 10, sequence_length))
    y_train = np.zeros((num_cars, len(train_indices) - 10, sequence_length))
    x_test = np.zeros((num_cars, len(test_indices) - 10, 2))
    y_test = np.zeros((num_cars, len(test_indices) - 10, 2))

    # Prepare data for LSTM
    for i in range(num_cars):
        for j in range(10, len(train_indices)):
            x_train[i, j - 10, 0] = x_traj[i, train_indices[j - 10]]
            x_train[i, j - 10, 1] = y_traj[i, train_indices[j - 10]]
            y_train[i, j - 10, 0] = x_traj[i, train_indices[j]]
            y_train[i, j - 10, 1] = y_traj[i, train_indices[j]]

        for j in range(10, len(test_indices)):
            x_test[i, j - 10, 0] = x_traj[i, test_indices[j - 10]]
            x_test[i, j - 10, 1] = y_traj[i, test_indices[j - 10]]
            y_test[i, j - 10, 0] = x_traj[i, test_indices[j]]
            y_test[i, j - 10, 1] = y_traj[i, test_indices[j]]

    return x_train, y_train, x_test, y_test


# Train LSTM model
import matplotlib.pyplot as plt

def train_model(model, x_train, y_train, lr=0.01, num_epochs=100):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Reshape data to fit LSTM input shape
        input = torch.tensor(x_train).float().transpose(0, 1).to(device)
        target = torch.tensor(y_train[:, -1, :]).float().to(device)

        # Forward pass
        output = model(input)

        # Backward pass
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')

            # Visualize actual and predicted trajectories for first car in test set
            plt.figure(figsize=(12, 4))
            plt.scatter(x_test[0, :, 0], x_test[0, :, 1], c='orange', s=10)
            plt.plot(y_test[0, :, 0], y_test[0, :, 1], c='orange')
            plt.plot(output[0, :, 0].detach().cpu().numpy(), output[0, :, 1].detach().cpu().numpy(), c='blue')
            plt.title(f'Actual and predicted trajectories for car {i} in test set (epoch {epoch})')
            plt.xlabel('x_utm')
            plt.ylabel('y_utm')
            plt.show()


# Main function
if __name__ == '__main__':
    x_train, y_train, x_test, y_test = generate_data(num_cars=5, traj_length=3000)

    input_size = 2  # x_utm, y_utm
    hidden_size = 16
    output_dim = 2  # x_utm, y_utm
    model = LSTM(input_size=input_size, hidden_size=hidden_size, output_dim=output_dim)

    train_model(model, x_train, y_train, lr=0.001, num_epochs=500)