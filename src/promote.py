import yaml
from mlflow import tracking

def promote_model():
    """
    Evaluates the latest registered models. If a model meets the required 
    F1 score threshold from the config, it is promoted to 'Production'.
    """
    # Load config for thresholds
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    model_name = config['registry']['model_name']
    min_f1 = config['registry']['min_f1_score']
    
    client = tracking.MlflowClient()
    
    try:
        # Get all versions of the model
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as e:
        print(f"❌ Error fetching model: {e}. Has it been registered yet?")
        return
    
    if not versions:
        print("No models found in the registry.")
        return
        
    # Get the latest version
    latest_version = sorted(versions, key=lambda v: v.version, reverse=True)[0]
    
    # Retrieve the metrics logged during that specific model's run
    run = client.get_run(latest_version.run_id)
    run_f1 = run.data.metrics.get("f1_score", 0.0)
    
    print(f"Evaluating Model Version: {latest_version.version}")
    print(f"Run ID: {latest_version.run_id}")
    print(f"Achieved F1: {run_f1:.4f} | Required F1: {min_f1:.4f}")
    
    if run_f1 >= min_f1:
        # Promote to Production!
        print(f"🏆 Policy Passed! Promoting Version {latest_version.version} to Production.")
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version.version,
            stage="Production",
            archive_existing_versions=True # Automatically demotes old prod models
        )
    else:
        print(f"⚠️ Policy Failed! Model did not meet minimum F1 score requirement.")

if __name__ == "__main__":
    promote_model()