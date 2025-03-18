import torch
import torch.nn as nn
import torch.nn.functional as F

class MyMethod(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(MyMethod, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Flatten the input tensor
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def build_model(params):
    """
    Build and return an instance of MyMethod based on specified hyperparameters.
    
    Parameters:
        params (dict): Dictionary containing hyperparameters such as:
            - input_dim: Dimension of the input data.
            - hidden_dim: Number of neurons in the hidden layer.
            - output_dim: Number of classes for prediction.
    
    Returns:
        model (MyMethod): Instance of the MyMethod model.
    """
    input_dim = params.get("input_dim", 784)
    hidden_dim = params.get("hidden_dim", 128)
    output_dim = params.get("output_dim", 10)
    model = MyMethod(input_dim, hidden_dim, output_dim)
    return model

if __name__ == "__main__":
    # Example usage: build the model using default parameters or those provided by a config.
    params = {"input_dim": 784, "hidden_dim": 128, "output_dim": 10}
    model = build_model(params)
    print(model)