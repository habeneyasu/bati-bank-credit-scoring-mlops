# MLflow UI Navigation Guide

## Accessing MLflow UI

1. Start MLflow UI:
   ```bash
   mlflow ui
   # Or: ./scripts/start_mlflow_ui.sh
   ```

2. Open in browser: http://localhost:5000

## Viewing Experiments

### 1. Experiment List
- **Location**: Main page or "Experiments" tab
- **Shows**: All experiments with creation date and description
- **Action**: Click on experiment name to view runs

### 2. Experiment Runs
- **Location**: After clicking an experiment
- **Shows**: 
  - All runs in the experiment
  - Key metrics (e.g., test_roc_auc, test_accuracy)
  - Parameters
  - Run status and duration
- **Actions**:
  - Click run name to view details
  - Select multiple runs to compare
  - Sort by any column (click column header)

### 3. Run Details
- **Location**: Click on a specific run
- **Shows**:
  - **Parameters**: All model hyperparameters
  - **Metrics**: Training and test metrics
  - **Artifacts**: Saved models, plots, datasets
  - **Tags**: Model type, training method, etc.
- **Actions**:
  - Download model artifacts
  - View model code
  - Register model (if not already registered)

## Comparing Runs

### Method 1: Compare Button
1. Go to experiment runs page
2. Select 2+ runs (checkboxes on left)
3. Click "Compare" button
4. View side-by-side comparison:
   - Parameter differences
   - Metric comparisons
   - Visualizations

### Method 2: Parallel Coordinates Plot
- Shows all runs in parallel coordinates
- Helps identify parameter patterns
- Filter by metric ranges

## Model Registry

### Viewing Registered Models
1. Click "Models" tab in MLflow UI
2. See all registered models:
   - Model name
   - Current stage (Staging/Production/Archived)
   - Latest version
   - Metrics and parameters

### Model Versions
1. Click on a registered model
2. View all versions:
   - Version number
   - Stage
   - Metrics
   - Creation date
3. Click version to see details

### Transitioning Stages
1. Go to model version details
2. Click "Stage" dropdown
3. Select new stage:
   - **Staging**: Testing before production
   - **Production**: Live model
   - **Archived**: Deprecated models

## Key Metrics to Monitor

### For Credit Scoring Models
- **test_roc_auc**: Primary metric (higher is better)
- **test_accuracy**: Overall accuracy
- **test_precision**: Precision for high-risk class
- **test_recall**: Recall for high-risk class
- **test_f1_score**: F1 score for high-risk class

### Training vs Test Metrics
- Compare `train_*` vs `test_*` metrics
- Large gap indicates overfitting
- Similar values indicate good generalization

## Tips for Effective Comparison

1. **Sort by Primary Metric**: Click "test_roc_auc" column header
2. **Filter Runs**: Use search/filter to find specific runs
3. **Tag Your Runs**: Use tags to categorize (e.g., "baseline", "tuned", "production")
4. **Document Changes**: Use run descriptions to note what changed

## Common Workflows

### Finding Best Model
1. Go to experiment runs
2. Sort by `test_roc_auc` (descending)
3. Top run is your best model
4. Click to view details
5. Register if ready for production

### Comparing Tuning Methods
1. Filter runs by tag (e.g., "tuning_method: grid_search")
2. Compare with "tuning_method: random_search"
3. Identify which method found better parameters

### Tracking Model Evolution
1. View all versions of a registered model
2. Compare metrics across versions
3. Track performance improvements/regressions

---

**Quick Reference**:
- **Experiments**: List of all experiments
- **Runs**: Individual training runs within an experiment
- **Models**: Registered models in model registry
- **Artifacts**: Saved files (models, plots, etc.)

