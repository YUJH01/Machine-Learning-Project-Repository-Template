import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineA(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super(BaselineA, self).__init__()
        # A simple one-layer linear classifier as a baseline model
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        # Flatten the input tensor
        x = x.view(x.size(0), -1)
        return self.linear(x)

def build_model(params):
    """
    Build and return an instance of BaselineA based on specified hyperparameters.
    
    Parameters:
        params (dict): Dictionary containing hyperparameters such as:
            - input_dim: Dimension of the input data.
            - output_dim: Number of classes for prediction.
            
    Returns:
        model (BaselineA): Instance of the BaselineA model.
    """
    input_dim = params.get("input_dim", 784)
    output_dim = params.get("output_dim", 10)
    model = BaselineA(input_dim, output_dim)
    return model

if __name__ == "__main__":
    # Example usage: build the model using default parameters or those provided by a configuration.
    params = {"input_dim": 784, "output_dim": 10}
    model = build_model(params)
    print(model)