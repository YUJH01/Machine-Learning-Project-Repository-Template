import importlib

def load_model(model_config):
    """
    Dynamically load and build a model based on configuration.
    
    Parameters:
        model_config (dict): Configuration dictionary with keys:
            - name: Name of the model (e.g., 'my_method', 'baseline_a', 'baseline_b')
            - params: Dictionary containing the model hyperparameters.
            
    Returns:
        model: An instance of the requested model.
    """
    model_name = model_config.get("name")
    model_params = model_config.get("params", {})

    if not model_name:
        raise ValueError("No model name provided in configuration.")

    # Construct the module path based on model name.
    module_name = f"src.models.{model_name}"
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_name}'.") from e

    if not hasattr(module, "build_model"):
        raise AttributeError(f"Module '{module_name}' does not have a build_model function.")

    model = module.build_model(model_params)
    return model

if __name__ == "__main__":
    # Example configuration usage
    config = {
        "name": "my_method",
        "params": {
            "input_dim": 784,
            "hidden_dim": 128,
            "output_dim": 10
        }
    }
    model = load_model(config)
    print(model)