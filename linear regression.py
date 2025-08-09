#@ Load dependencies
# general
import io
# data
import numpy as np
import pandas as pd
#machine learning
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
# data visualization
import plotly.express as px
from matplotlib import pyplot as plt
from numpy.random import random_sample
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
from pyexpat import features

#@ dataset
chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")
# dataframe to sp columns
training_df = chicago_taxi_dataset.loc[:, ('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')]

# print(training_df.head())
# print(training_df.shape)
# print(training_df.isnull().sum())
#
# # View dataset statistics
# print(f'Total number of rows: {len(training_df.index)}\n\nTotal number of columns: {len(training_df.columns)}\n\n')
# training_df.describe(include='all')
#
# # max mean freq missing
# missing_values = training_df.isnull().sum()
# print("no" if missing_values.sum() == 0 else "yes")
# most_freq_payment_type = training_df['PAYMENT_TYPE'].value_counts().idxmax()
# print(most_freq_payment_type)
# num_unique_company = training_df['COMPANY'].nunique()
# print(num_unique_company)
# mean_distance = training_df['TIP_RATE'].mean()
# print(f'{mean_distance:.4f}')
# max_fare = training_df['FARE'].max()
# print(f'{max_fare:.2f}')
#
# # Correlation matrix
# training_df.corr(numeric_only=True)
# # Pairplot
# sns.pairplot(training_df, x_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"], y_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"])
# plt.show()
# px.scatter_matrix(training_df, dimensions=["FARE", "TRIP_MILES", "TRIP_SECONDS"])

# PyTorch Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# Custom dataset class
class TaxiDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features.values)
        self.labels = torch.FloatTensor(labels.values)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Train Model
def train_model(df, feature_names, label_name, learning_rate=0.01, epochs=1000, batch_size=32):
    # Prepare data
    X = df[feature_names]
    y = df[[label_name]]

    # Create dataset and dataloader
    dataset = TaxiDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss and optimizer
    model = LinearRegressionModel(len(feature_names))
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # For tracking loss
    epoch_list = []
    rmse_list = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in dataloader:
            # Forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate RMSE for the whole dataset
        with torch.no_grad():
            model.eval()
            y_pred = model(torch.FloatTensor(X.values))
            rmse = torch.sqrt(criterion(y_pred, torch.FloatTensor(y.values)))

            epoch_list.append(epoch)
            rmse_list.append(rmse.item())

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], RMSE: {rmse.item():.4f}')

    # Extract weights and bias
    weights = model.linear.weight.detach().numpy().T
    bias = model.linear.bias.detach().numpy()

    return weights, bias, epoch_list, rmse_list

# Make plots
def make_plots(df, feature_names, label_name, model_output, sample_size=200):
    random_sample = df.sample(n=sample_size).copy()
    random_sample.reset_index(drop=True, inplace=True)
    weights, bias, epochs, rmse = model_output

    is_2d_plot = len(feature_names) == 1
    model_plot_type = "scatter" if is_2d_plot else "surface"
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Loss Curve", "Model Plot"),
                        specs=[[{"type": "scatter"}, {"type": model_plot_type}]])
    plot_data(random_sample, feature_names, label_name, fig)
    plot_model(random_sample, feature_names, weights, bias, fig)
    plot_loss_curve(epochs, rmse, fig) #Shows how model error decreases during training

    fig.show()
    return
def plot_loss_curve(epochs, rmse, fig):
    curve = px.line(x=epochs, y=rmse)
    curve.update_traces(line_color="lightgreen", line_width=3)

    fig.append_trace(curve.data[0], row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(
        title_text="RMSE", 
        row=1, 
        col=1, 
        range=[min(rmse) * 0.8, max(rmse)]  # Fixed to use min() and max()
    )

    return
def plot_data(df, feature_names, label_name, fig):
    if len(feature_names) == 1:
        scatter = px.scatter(df, x=feature_names[0], y=label_name)
    else:
        scatter = px.scatter_3d(df, x=feature_names[0], y=feature_names[1], z=label_name)

    fig.append_trace(scatter.data[0], row=1, col=2)
    if len(feature_names) == 1:
        fig.update_xaxes(title_text=feature_names[0], row=1, col=2)
        fig.update_yaxes(title_text=label_name, row=1, col=2)
    else:
        fig.update_layout(scene1=dict(xaxis_title=feature_names[0], yaxis_title=feature_names[1], zaxis_title=label_name))

    return

def plot_model(df, feature_names, weights, bias, fig):
    df['FARE_PREDICTED'] = bias[0]
    for index, feature in enumerate(feature_names):
        df['FARE_PREDICTED'] = df['FARE_PREDICTED'] + weights[index][0] * df[feature] #Prediction = bias + weight₁ × feature₁ + weight₂ × feature₂ + ..., weight[index][0] gives scalar value instead of vector

    if len(feature_names) == 1:
        model = px.line(df, x=feature_names[0], y='FARE_PREDICTED')
        model.update_traces(line_color="skyblue", line_width=3)
        fig.append_trace(model.data[0], row=1, col=2)
    else:
        # Create surface plot for 3D
        x_range = np.linspace(df[feature_names[0]].min(), df[feature_names[0]].max(), 20)
        y_range = np.linspace(df[feature_names[1]].min(), df[feature_names[1]].max(), 20)
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        z_grid = bias[0] + weights[0][0] * x_grid + weights[1][0] * y_grid

        surface = go.Surface(x=x_grid, y=y_grid, z=z_grid, opacity=0.7, colorscale='blues')
        fig.add_trace(surface, row=1, col=2)

    return

def model_info(feature_names, label_name, model_output):
    weights = model_output[0]
    bias = model_output[1]

    n1 = "\n"
    header = "-"* 80
    banner = f"{header}\n{"Model Information".center(78)}\n{header}"

    info = ""
    equation = label_name+" = "

    for index, feature in enumerate(feature_names):
        equation += f"{weights[index][0]:.4f} * {feature} + "
        info += f"Weight {index+1}: {weights[index][0]:.4f}\n"

    equation += f"{bias[0]:.4f}"
    info += f"Bias: {bias[0]:.4f}\n"

    info += f"Equation: {equation}\n"

    return banner + n1 + info + n1 + header

print("SUCCESS: defining plotting functions completed")

# Experiment 1
# the following variables are the hyperparameters
feature_names = ["TRIP_MILES"]
label_name = "FARE"
learning_rate = 0.001
epochs = 20
batch_size = 50

# Create and train the model with one feature
print("\nRunning experiment: one_feature")
model_output_1 = train_model(
    df=training_df,
    feature_names=feature_names,
    label_name=label_name,
    learning_rate=learning_rate,
    epochs=epochs,
    batch_size=batch_size
)

# Display model information
print(model_info(feature_names, label_name, model_output_1))

# Plot the results
make_plots(training_df, feature_names, label_name, model_output_1)

# Experiment 3
# Add TRIP_MINUTES feature derived from TRIP_SECONDS
training_df['TRIP_MINUTES'] = training_df['TRIP_SECONDS'] / 60

# The following variables are the hyperparameters
feature_names_3 = ['TRIP_MILES', 'TRIP_MINUTES']
label_name_3 = 'FARE'
learning_rate_3 = 0.001
epochs_3 = 20
batch_size_3 = 50

# Create and train the model with two features
print("\nRunning experiment: two_features")
model_output_3 = train_model(
    df=training_df,
    feature_names=feature_names_3,
    label_name=label_name_3,
    learning_rate=learning_rate_3,
    epochs=epochs_3,
    batch_size=batch_size_3
)

# Display model information
print(model_info(feature_names_3, label_name_3, model_output_3))

# Plot the results
make_plots(training_df, feature_names_3, label_name_3, model_output_3)

