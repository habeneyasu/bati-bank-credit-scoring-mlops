"""
Register Best Model in MLflow Model Registry

This script identifies the best model from all MLflow experiments and registers it
in the MLflow Model Registry for deployment.

This completes the model registration step of Task 5.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.mlflow_tracking import MLflowTracker
import mlflow


def main():
    """Main function to identify and register best model."""
    
    print("=" * 100)
    print("Identify and Register Best Model in MLflow Model Registry")
    print("=" * 100)
    print()
    
    # Initialize MLflow tracker
    tracker = MLflowTracker(
        experiment_name="credit_scoring",
        tracking_uri="file:./mlruns"
    )
    print()
    
    # Get best model from experiment
    print("Searching for best model...")
    print("-" * 100)
    print()
    
    try:
        best_run = tracker.get_best_run(metric='test_roc_auc')
        
        print("Best Model Found:")
        print(f"  Run Name: {best_run['run_name']}")
        print(f"  Run ID: {best_run['run_id']}")
        print(f"  Metric: {best_run['metric_name']}")
        print(f"  Score: {best_run['metric_value']:.4f}")
        print()
        
        # Display key parameters
        print("Model Parameters:")
        key_params = ['n_estimators', 'max_depth', 'min_samples_split', 
                     'C', 'penalty', 'solver', 'criterion']
        for param in key_params:
            if param in best_run['params']:
                print(f"  {param}: {best_run['params'][param]}")
        print()
        
        # Check if model is already registered
        print("Checking Model Registry...")
        print("-" * 100)
        
        try:
            # Try to get existing registered model
            registered_models = mlflow.search_registered_models()
            model_exists = False
            existing_model_name = None
            
            for model in registered_models:
                if model.name == "credit_scoring_model":
                    model_exists = True
                    existing_model_name = model.name
                    print(f"Found existing registered model: {existing_model_name}")
                    print(f"  Latest version: {model.latest_versions[0].version}")
                    print(f"  Current stage: {model.latest_versions[0].current_stage}")
                    break
            
            if not model_exists:
                print("No existing registered model found. Will create new registration.")
            print()
            
        except Exception as e:
            print(f"Could not check existing models: {e}")
            print("Will proceed with registration...")
            print()
        
        # Register the best model
        print("Registering best model...")
        print("-" * 100)
        
        model_name = "credit_scoring_model"
        
        try:
            model_version = tracker.register_model(
                run_id=best_run['run_id'],
                model_name=model_name,
                artifact_path="model",
                stage="Staging"  # Start in Staging, can promote to Production
            )
            
            print()
            print("=" * 100)
            print("Model Successfully Registered!")
            print("=" * 100)
            print()
            print(f"Model Name: {model_name}")
            print(f"Version: {model_version}")
            print(f"Stage: Staging")
            print(f"Run ID: {best_run['run_id']}")
            print(f"Performance: {best_run['metric_name']} = {best_run['metric_value']:.4f}")
            print()
            
            # Ask if user wants to promote to Production
            print("Would you like to promote this model to Production stage?")
            print("(This can also be done later in MLflow UI)")
            print()
            print("To promote manually:")
            print("  1. Go to MLflow UI: http://localhost:5000")
            print("  2. Click 'Models' tab")
            print("  3. Click on 'credit_scoring_model'")
            print("  4. Select the version and change stage to 'Production'")
            print()
            
            # Optionally promote to Production
            promote = input("Promote to Production now? (y/n): ").strip().lower()
            
            if promote == 'y':
                try:
                    tracker.transition_model_stage(
                        model_name=model_name,
                        version=model_version,
                        stage="Production"
                    )
                    print()
                    print("✓ Model promoted to Production stage!")
                except Exception as e:
                    print(f"Could not promote to Production: {e}")
                    print("You can promote it manually in MLflow UI")
            else:
                print("Model remains in Staging stage. You can promote it later.")
            
            print()
            
        except Exception as e:
            print(f"Error registering model: {e}")
            print()
            print("Troubleshooting:")
            print("  1. Ensure MLflow is properly installed")
            print("  2. Check that the run exists and has a model artifact")
            print("  3. Try registering manually in MLflow UI")
            return
        
        # Display model registry information
        print("=" * 100)
        print("Model Registry Information")
        print("=" * 100)
        print()
        print("To view registered model:")
        print("  1. Open MLflow UI: http://localhost:5000")
        print("  2. Click 'Models' tab")
        print("  3. Click on 'credit_scoring_model'")
        print("  4. View all versions and their stages")
        print()
        print("To use registered model in code:")
        print("""
        import mlflow
        
        # Load model from registry
        model = mlflow.sklearn.load_model(
            model_uri="models:/credit_scoring_model/Production"
        )
        
        # Or load specific version
        model = mlflow.sklearn.load_model(
            model_uri="models:/credit_scoring_model/1"
        )
        """)
        print()
        
    except Exception as e:
        print(f"Error finding best model: {e}")
        print()
        print("Possible issues:")
        print("  1. No experiments found - run training script first:")
        print("     python examples/train_with_mlflow.py")
        print("  2. Experiment name mismatch")
        print("  3. No runs with test_roc_auc metric")
        return
    
    print("=" * 100)
    print("Model Registration Complete!")
    print("=" * 100)


if __name__ == "__main__":
    main()

