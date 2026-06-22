import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report

def get_git_commit():
    """Attempts to get the current git commit hash for tracking."""
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "untracked-local"

def train():
    # 1. Load Configuration
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # 2. Load Data
    print("Loading data...")
    df = pd.read_csv(config['data']['raw_path'])
    X = df['text']
    y = df['category']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['training']['test_size'], 
        random_state=config['training']['random_state']
    )
    
    # 3. Create Scikit-Learn Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=config['training']['tfidf_max_features'])),
        ('rf', RandomForestClassifier(
            n_estimators=config['training']['rf_n_estimators'],
            max_depth=config['training']['rf_max_depth'],
            random_state=config['training']['random_state']
        ))
    ])
    
    # 4. MLflow Tracking Run
    # This sets the experiment name. If it doesn't exist, MLflow creates it.
    mlflow.set_experiment("ticket_classification")
    
    with mlflow.start_run() as run:
        print(f"Started MLflow Run: {run.info.run_id}")
        
        # Log Configuration Parameters (Hyperparameters)
        mlflow.log_params(config['training'])
        
        # Log Source Code Commit (Governance & Reproducibility)
        mlflow.set_tag("commit_hash", get_git_commit())
        mlflow.set_tag("developer", "mlops-engineer")
        
        # Train Model
        print("Training model...")
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Metrics -> F1 Score: {f1:.4f} | Accuracy: {acc:.4f}")
        
        # Log Metrics
        mlflow.log_metrics({
            "f1_score": f1,
            "accuracy": acc
        })
        
        # 5. Model Registry
        # Log the model and instantly register it as a new version
        model_name = config['registry']['model_name']
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=model_name
        )
        print(f"✅ Model registered in MLflow under name: '{model_name}'")

if __name__ == "__main__":
    train()