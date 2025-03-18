import os
import json
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.models.model_factory import load_model

def load_config(config_path="experiments/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def get_dummy_data(n_samples=1000, input_dim=784, num_classes=10):
    """
    Generate dummy data for evaluation.
    Replace this with the logic for loading your actual test dataset.
    """
    x = torch.randn(n_samples, input_dim)
    y = torch.randint(0, num_classes, (n_samples,))
    return TensorDataset(x, y)

def evaluate():
    # Load configuration
    config = load_config()
    
    # Get model configuration (using the first model as default)
    model_config = config["models"][0]
    model = load_model(model_config)
    
    # Get evaluation configuration
    evaluation_config = config.get("evaluation", {})
    report_path = evaluation_config.get("report_path", "results/evaluation_report.json")
    
    # Prepare the test dataset (replace with actual test data loading)
    input_dim = model_config["params"].get("input_dim", 784)
    batch_size = model_config["params"].get("batch_size", 32)
    dataset = get_dummy_data(n_samples=1000, input_dim=input_dim, num_classes=10)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    # Define a loss function for evaluation
    criterion = nn.CrossEntropyLoss()
    
    # Set the model to evaluation mode
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples if total_samples else 0.0
    
    # Print evaluation metrics
    print(f"Evaluation Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Save evaluation report
    report = {
        "loss": avg_loss,
        "accuracy": accuracy,
    }
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Evaluation report saved to {report_path}")

if __name__ == "__main__":
    evaluate()