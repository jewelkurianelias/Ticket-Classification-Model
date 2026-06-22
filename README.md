# Reproducible Experiment Tracking & Model Registry

Welcome to the third MLOps project! Here, we solve the problem of "lost models" by strictly tracking our experiments and managing a Model Registry.

## 📁 Project Structure

```text
.
├── configs/
│   └── config.yaml        # Centralized YAML configuration for reproducible training.
├── src/
│   ├── generate_data.py   # Script to generate synthetic support tickets.
│   ├── train.py           # MLflow training script for the classification model.
│   └── promote.py         # Script to evaluate and promote models to Production.
├── tests/
│   └── test_train.py      # Basic unit tests for data and training logic.
├── .gitignore             # Specifies intentionally untracked files to ignore.
├── README.md              # Project documentation and quickstart guide.
├── REPORT.md              # Documentation of model governance and selection.
└── requirements.txt       # Python dependencies for the project.
```

## 🚀 Quickstart

### 1. Set up the environment

```bash
python -m venv venv
source venv/bin/activate  # (Windows: .\venv\Scripts\activate)
pip install -r requirements.txt
```

### 2. Create a local git repo (Required for commit tracking!)

```bash
git init
git add .
git commit -m "Initial commit for experiment tracking"
```

### 3. Generate Dummy Data & Run Tests

```bash
python src/generate_data.py
python -m pytest tests/
```

### 4. Train the Model & Log to MLflow

```bash
python src/train.py
```

> **Note:** This will automatically register the model under the name 'TicketClassifier' in your local registry.

### 5. Promote the Model to Production

```bash
python src/promote.py
```

*If the model's F1 score exceeds the limit set in `configs/config.yaml`, it will be officially tagged as Production!*

### 6. View the Dashboard

```bash
mlflow ui
```

Open http://127.0.0.1:5000 in your browser. Click on **Experiments** to see your parameters and metrics. Click on **Models** at the top to see your Model Registry and verify your model is marked as "Production"!
