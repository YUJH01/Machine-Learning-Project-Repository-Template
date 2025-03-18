# 📘 Machine Learning Method Repository

This repository presents a novel machine learning method, accompanied by benchmark models for performance comparison.

---

## 🚀 Project Structure

```plaintext
📁 src
│
├── 📁 models               # Contains different model architectures
│   ├── my_method.py       # Your proposed method
│   ├── baseline_a.py      # Benchmark model A
│   ├── baseline_b.py      # Benchmark model B
│   └── model_factory.py   # Dynamically loads models based on config
│
└── 📁 datasets             # Data loading and preprocessing scripts
    ├── custom_dataset.py
    └── data_preprocessing.py   # Script to preprocess raw data

📁 experiments              # Configurations and results
│
└── config.yaml             # YAML file specifying models and hyperparameters

📁 results
└── checkpoints            # Saved model weights

📁 notebooks               # Jupyter notebooks for analysis
└── analysis.ipynb         # Visualization and analysis of results

📁 scripts                 # Bash scripts for running experiments
📁 slurm_scripts           # SLURM scripts for HPC benchmarking
📁 logs                   # Log files for experiments and HPC runs

train.py                   # Unified training script
evaluation.py              # Evaluation script for computing metrics
requirements.txt           # Package dependencies
README.md
```

---

## 🔧 Setup

1. Clone the repo:
    ```bash
    git clone https://github.com/your-repo/ml-new-method.git
    cd ml-new-method
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Prepare the dataset (modify `datasets/custom_dataset.py` and `datasets/data_preprocessing.py` if needed).

---

## 🔄 Data Preprocessing

Place your raw data files (for example, `raw_data.csv`) in the `datasets/` folder. Run the data preprocessing script to clean and prepare your data:

```bash
python datasets/data_preprocessing.py
```

This script performs basic cleaning (e.g., handling missing values) and saves the processed data (e.g., as `data.csv`) which is then used by the rest of the project.

---

## 📥 Loading the Data

To load your dataset, ensure your data files are placed in `datasets/`. Modify `datasets/custom_dataset.py` to define how the data should be loaded and preprocessed.

Example usage in your script:

```python
from src.datasets.custom_dataset import load_data

train_data, val_data, test_data = load_data("datasets/data.csv")
```

Ensure the data format matches your model's expectations. If using images, adjust accordingly.

---

## 🛠️ Training

Run training with:
```bash
python train.py
```

The model and dataset are chosen based on `experiments/config.yaml`. You can add new models and configurations easily.

---

## 📊 Evaluation

After training, run the evaluation script to compute metrics (e.g., accuracy, F1-score) on the test set. Execute:
```bash
python evaluation.py
```

The evaluation script loads the trained model and dataset as specified in `experiments/config.yaml`, prints evaluation metrics to the console, and saves detailed results to the `results/` directory.

---

## 🔬 Benchmarking

This repository supports comparing your proposed method against benchmark models.

1. Define benchmark models in `src/models/`.
2. Specify each model and its hyperparameters in `experiments/config.yaml`.
3. Run training and evaluation for each model.
4. Results (e.g., accuracy, loss) are saved to the `results/` directory.

Example configuration in `config.yaml`:
```yaml
models:
  - name: my_method
    params:
      learning_rate: 0.001
      epochs: 100
  - name: baseline_a
    params:
      learning_rate: 0.001
      epochs: 100
```

The `model_factory.py` script ensures the correct model is loaded from the configuration.

---

## ⚡ Running a Bash Script with nohup from the Terminal

If you have a bash script (for example, `scripts/run_experiment.sh`) that you want to run in the background so it continues executing after you log out, you can use `nohup`.

Run the script with:

```bash
nohup bash scripts/run_experiment.sh > logs/run_experiment.log 2>&1 &
```

This command will:
- Execute `run_experiment.sh` from the `scripts` directory in the background.
- Redirect both standard output and error to `logs/run_experiment.log` so you can review the logs later.
- Allow the process to keep running even if you close your terminal.

---

## 🖥️ HPC Benchmarking with SLURM

For HPC environments using the SLURM workload manager, you can benchmark models using the provided SLURM script. The script is located at `slurm_scripts/slurm_run_model.slurm`.

To run a single model on the HPC cluster:

1. Edit the SLURM script if necessary (e.g., adjust resource parameters or virtual environment path).
2. Submit the job using:
    ```bash
    sbatch slurm_scripts/slurm_run_model.slurm
    ```

This will run the training for the specified model (`my_method`) with the configuration defined in `experiments/config.yaml`.

---

## ✅ TODO List

### Core Functionality
- [x] Implement your proposed method
- [x] Add benchmark models
- [x] Create a unified training script
- [x] Implement dynamic model loading
- [x] Build a flexible configuration system
- [x] Add dataset loading
- [x] Implement performance evaluation
- [ ] Write a data preprocessing script
- [ ] Write an evaluation script with metrics like accuracy, F1, etc.
- [ ] Create a results analysis Jupyter notebook
- [ ] Document all files thoroughly
- [ ] Add ablation study script

### Enhancements
- [ ] Add logging (TensorBoard/WandB)
- [ ] Implement learning rate scheduling
- [ ] Save checkpoints properly and handle resume training
- [ ] Add command-line argument parsing for configs
- [ ] Implement hyperparameter tuning automation

### Future Improvements
- [ ] Support more datasets
- [ ] Add visualization tools for model comparisons
- [ ] Implement distributed training support
- [ ] Add ensemble method support

---

## 💡 Contributing

Feel free to submit pull requests for improvements and new benchmarks!

---

## 🏅 License

MIT License