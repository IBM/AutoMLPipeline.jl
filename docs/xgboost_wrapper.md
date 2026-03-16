# XGBoost Wrapper for AutoMLPipeline.jl

This document describes the XGBoost wrapper implementation for AutoMLPipeline.jl, which provides a consistent interface similar to SKLearners.jl.

## Overview

The XGBoost wrapper allows you to use XGBoost models within the AutoMLPipeline framework with the standard `fit!` and `transform!` functions. It supports all major XGBoost learners including classifiers, regressors, and random forest variants.

## Installation Requirements

Before using the XGBoost wrapper, ensure you have XGBoost installed in your Python environment:

```bash
pip install xgboost
```

Or using conda:

```bash
conda install -c conda-forge xgboost
```

## Available Learners

The wrapper provides access to the following XGBoost learners:

- **XGBClassifier**: Gradient boosting classifier
- **XGBRegressor**: Gradient boosting regressor
- **XGBRFClassifier**: Random forest classifier (XGBoost implementation)
- **XGBRFRegressor**: Random forest regressor (XGBoost implementation)
- **XGBRanker**: Learning to rank model

## Basic Usage

### Classification Example

```julia
using AutoMLPipeline
using DataFrames

# Load data
iris = getiris()
X = iris[:, 1:4]
y = iris[:, 5] |> Vector

# Split data
(train_X, train_y), (test_X, test_y) = train_test_split(X, y, 0.7)

# Create and train XGBoost classifier
xgb = XGBoostLearner("XGBClassifier")
fit!(xgb, train_X, train_y)

# Make predictions
predictions = transform!(xgb, test_X)
```

### Regression Example

```julia
# Create XGBoost regressor with custom parameters
xgb_reg = XGBoostLearner("XGBRegressor"; 
                         max_depth=5,
                         learning_rate=0.05,
                         n_estimators=200)

# Fit and predict
fit!(xgb_reg, train_X, train_y)
predictions = transform!(xgb_reg, test_X)
```

## Constructor Signatures

### XGBoostLearner

```julia
# Basic constructor with default parameters
XGBoostLearner()

# Constructor with learner name
XGBoostLearner(learner::String)

# Constructor with learner name and dictionary of arguments
XGBoostLearner(learner::String, args::Dict)

# Constructor with learner name and keyword arguments
XGBoostLearner(learner::String; kwargs...)
```

## Common Parameters

XGBoost models support numerous hyperparameters. Here are the most commonly used:

### Tree Booster Parameters

- `max_depth` (default: 6): Maximum depth of a tree
- `learning_rate` or `eta` (default: 0.3): Step size shrinkage to prevent overfitting
- `n_estimators` (default: 100): Number of boosting rounds
- `min_child_weight` (default: 1): Minimum sum of instance weight needed in a child
- `gamma` (default: 0): Minimum loss reduction required to make a split
- `subsample` (default: 1): Subsample ratio of the training instances
- `colsample_bytree` (default: 1): Subsample ratio of columns when constructing each tree
- `colsample_bylevel` (default: 1): Subsample ratio of columns for each level
- `colsample_bynode` (default: 1): Subsample ratio of columns for each node

### Regularization Parameters

- `reg_alpha` (default: 0): L1 regularization term on weights
- `reg_lambda` (default: 1): L2 regularization term on weights

### Learning Task Parameters

- `objective`: Learning objective (auto-detected based on learner type)
  - Classification: 'binary:logistic', 'multi:softmax', 'multi:softprob'
  - Regression: 'reg:squarederror', 'reg:logistic', 'reg:gamma'
- `eval_metric`: Evaluation metric
- `seed` or `random_state`: Random seed for reproducibility

### Other Parameters

- `booster` (default: 'gbtree'): Which booster to use ('gbtree', 'gblinear', 'dart')
- `tree_method` (default: 'auto'): Tree construction algorithm ('auto', 'exact', 'approx', 'hist', 'gpu_hist')
- `n_jobs` (default: 1): Number of parallel threads
- `verbosity` (default: 1): Verbosity of printing messages (0=silent, 1=warning, 2=info, 3=debug)

## Advanced Usage

### Using with Pipelines

```julia
using AutoMLPipeline: @pipeline

# Create pipeline components
imputer = Imputer()
scaler = SKPreprocessor("StandardScaler")
xgb = XGBoostLearner("XGBClassifier"; max_depth=3, n_estimators=100)

# Create pipeline
pipeline = @pipeline imputer |> scaler |> xgb

# Fit and transform
fit!(pipeline, train_X, train_y)
predictions = transform!(pipeline, test_X)
```

### Using xgboostoperator Helper

```julia
# Create XGBoost learner using helper function
xgb = xgboostoperator("XGBClassifier"; max_depth=4, n_estimators=150)
fit!(xgb, train_X, train_y)
predictions = transform!(xgb, test_X)
```

### Listing Available Learners

```julia
# Display all available XGBoost learners
xgboostlearners()
```

## API Reference

### Functions

#### `fit!(xgb::XGBoostLearner, X::DataFrame, y::Vector)::Nothing`

Trains the XGBoost model on the provided data.

**Arguments:**
- `xgb`: XGBoostLearner instance
- `X`: Training features as DataFrame
- `y`: Training labels as Vector

**Returns:** Nothing (modifies xgb in-place)

#### `fit(xgb::XGBoostLearner, X::DataFrame, y::Vector)::XGBoostLearner`

Trains the XGBoost model and returns a copy.

**Arguments:**
- `xgb`: XGBoostLearner instance
- `X`: Training features as DataFrame
- `y`: Training labels as Vector

**Returns:** A new XGBoostLearner instance with trained model

#### `transform!(xgb::XGBoostLearner, X::DataFrame)::Vector`

Makes predictions using the trained model.

**Arguments:**
- `xgb`: Trained XGBoostLearner instance
- `X`: Features as DataFrame

**Returns:** Vector of predictions (String for classification, Float64 for regression)

#### `transform(xgb::XGBoostLearner, X::DataFrame)::Vector`

Makes predictions (non-mutating version).

**Arguments:**
- `xgb`: Trained XGBoostLearner instance
- `X`: Features as DataFrame

**Returns:** Vector of predictions

#### `xgboostlearners()`

Displays all available XGBoost learners and usage information.

#### `xgboostoperator(name::String; args...)::Machine`

Helper function to create XGBoost learners.

**Arguments:**
- `name`: Name of the XGBoost learner
- `args...`: Keyword arguments for the learner

**Returns:** XGBoostLearner instance

## Examples

See `examples/xgboost_example.jl` for comprehensive examples including:
- Basic classification and regression
- Custom parameter tuning
- Random forest variants
- Pipeline integration
- Cross-validation

## Comparison with SKLearners

The XGBoost wrapper follows the same design pattern as SKLearners:

| Feature | SKLearners | XGBoostLearners |
|---------|-----------|-----------------|
| Interface | `fit!`, `transform!` | `fit!`, `transform!` |
| Python backend | scikit-learn | xgboost |
| Constructor | `SKLearner(name; args...)` | `XGBoostLearner(name; args...)` |
| Helper function | `skoperator(name; args...)` | `xgboostoperator(name; args...)` |
| List learners | `sklearners()` | `xgboostlearners()` |

## Performance Tips

1. **Tree Method**: For large datasets, use `tree_method='hist'` or `tree_method='gpu_hist'` (if GPU available)
2. **Early Stopping**: Use `early_stopping_rounds` parameter to prevent overfitting
3. **Regularization**: Tune `reg_alpha` and `reg_lambda` to control model complexity
4. **Subsampling**: Use `subsample` and `colsample_bytree` to reduce overfitting
5. **Learning Rate**: Lower learning rates (0.01-0.1) with more estimators often work better
6. **Parallel Processing**: Set `n_jobs=-1` to use all CPU cores

## Troubleshooting

### XGBoost not found

If you get an error about XGBoost not being found:
```julia
# Install XGBoost in your Python environment
using Pkg
Pkg.add("PythonCall")
using PythonCall
# Then install xgboost via pip or conda
```

### Memory Issues

For large datasets:
- Use `tree_method='hist'` for memory-efficient histogram-based algorithm
- Reduce `max_depth` to limit tree complexity
- Use `subsample` to train on a subset of data

## References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [AutoMLPipeline.jl Documentation](https://github.com/IBM/AutoMLPipeline.jl)