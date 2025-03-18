import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.model_factory import load_model

def load_config(config_path="experiments/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def get_dummy_data(n_samples=1000, input_dim=784, num_classes=10):
    """
    Generate a dummy dataset for demonstration.
    Replace this with actual data loading logic (e.g., using a custom dataset).
    """
    x = torch.randn(n_samples, input_dim)
    y = torch.randint(0, num_classes, (n_samples,))
    return TensorDataset(x, y)

def train():
    # Load configuration from config.yaml
    config = load_config()

    # Get model configuration (assume the first model is used for training)
    model_config = config["models"][0]
    model = load_model(model_config)
    
    # Extract training parameters from configuration
    training_config = config.get("training", {})
    optimizer_name = training_config.get("optimizer", "adam")
    loss_function_name = training_config.get("loss_function", "cross_entropy")
    epochs = model_config["params"].get("epochs", 100)
    batch_size = model_config["params"].get("batch_size", 32)
    input_dim = model_config["params"].get("input_dim", 784)
    
    # Get dummy data (replace with custom dataset loading if needed)
    dataset = get_dummy_data(n_samples=1000, input_dim=input_dim, num_classes=10)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Define loss function
    if loss_function_name == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()  # Fallback to default
    
    # Define optimizer
    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters())
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    # Ensure the results directory exists (for saving models, logs, etc.)
    os.makedirs("results", exist_ok=True)
    train()