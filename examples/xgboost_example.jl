# XGBoost Wrapper Example
# This example demonstrates how to use the XGBoost wrapper in AutoMLPipeline.jl

using AutoMLPipeline
using DataFrames
using Statistics

println("=" ^ 70)
println("XGBoost Wrapper Examples for AutoMLPipeline.jl")
println("=" ^ 70)

# Example 1: Basic Classification with XGBoost
println("\n1. Basic XGBoost Classification")
println("-" ^ 70)

# Load iris dataset
iris = getiris()
X = iris[:, 1:4]
y = iris[:, 5] |> Vector

# Split data
(train_X, train_y), (test_X, test_y) = train_test_split(X, y, 0.7)

# Create XGBoost classifier with default parameters
xgb_clf = XGBoostLearner("XGBClassifier")

# Fit the model
fit!(xgb_clf, train_X, train_y)

# Make predictions
predictions = transform!(xgb_clf, test_X)

# Calculate accuracy
accuracy = sum(predictions .== test_y) / length(test_y)
println("Accuracy: $(round(accuracy * 100, digits=2))%")

# Example 2: XGBoost Classification with Custom Parameters
println("\n2. XGBoost Classification with Custom Parameters")
println("-" ^ 70)

# Create XGBoost classifier with custom parameters
xgb_clf_custom = XGBoostLearner("XGBClassifier"; 
                                max_depth=3,
                                learning_rate=0.1,
                                n_estimators=100,
                                subsample=0.8,
                                colsample_bytree=0.8)

# Fit and predict
fit!(xgb_clf_custom, train_X, train_y)
predictions_custom = transform!(xgb_clf_custom, test_X)

# Calculate accuracy
accuracy_custom = sum(predictions_custom .== test_y) / length(test_y)
println("Accuracy with custom parameters: $(round(accuracy_custom * 100, digits=2))%")

# Example 3: XGBoost Regression
println("\n3. XGBoost Regression")
println("-" ^ 70)

# Create synthetic regression data
n = 200
X_reg = DataFrame(
    feature1 = randn(n),
    feature2 = randn(n),
    feature3 = randn(n),
    feature4 = randn(n)
)
y_reg = 2.0 .* X_reg.feature1 .+ 3.0 .* X_reg.feature2 .- 
        1.5 .* X_reg.feature3 .+ 0.5 .* X_reg.feature4 .+ randn(n) .* 0.5

# Split data
(train_X_reg, train_y_reg), (test_X_reg, test_y_reg) = train_test_split(X_reg, y_reg, 0.7)

# Create XGBoost regressor
xgb_reg = XGBoostLearner("XGBRegressor"; 
                         max_depth=5,
                         learning_rate=0.05,
                         n_estimators=200)

# Fit and predict
fit!(xgb_reg, train_X_reg, train_y_reg)
predictions_reg = transform!(xgb_reg, test_X_reg)

# Calculate R² score
ss_res = sum((test_y_reg .- predictions_reg).^2)
ss_tot = sum((test_y_reg .- mean(test_y_reg)).^2)
r2 = 1 - ss_res / ss_tot
println("R² Score: $(round(r2, digits=4))")

# Calculate RMSE
rmse = sqrt(mean((test_y_reg .- predictions_reg).^2))
println("RMSE: $(round(rmse, digits=4))")

# Example 4: XGBoost Random Forest Classifier
println("\n4. XGBoost Random Forest Classifier")
println("-" ^ 70)

xgb_rf = XGBoostLearner("XGBRFClassifier"; 
                        max_depth=3,
                        n_estimators=50,
                        subsample=0.8)

fit!(xgb_rf, train_X, train_y)
predictions_rf = transform!(xgb_rf, test_X)

accuracy_rf = sum(predictions_rf .== test_y) / length(test_y)
println("XGBRFClassifier Accuracy: $(round(accuracy_rf * 100, digits=2))%")

# Example 5: Using xgboostoperator helper function
println("\n5. Using xgboostoperator Helper Function")
println("-" ^ 70)

xgb_op = xgboostoperator("XGBClassifier"; max_depth=4, n_estimators=150)
fit!(xgb_op, train_X, train_y)
predictions_op = transform!(xgb_op, test_X)

accuracy_op = sum(predictions_op .== test_y) / length(test_y)
println("Accuracy using xgboostoperator: $(round(accuracy_op * 100, digits=2))%")

# Example 6: Integration with AutoMLPipeline
println("\n6. Integration with AutoMLPipeline")
println("-" ^ 70)

# Create a pipeline with preprocessing and XGBoost
using AutoMLPipeline: @pipeline

# Create pipeline components
imputer = Imputer()
xgb_pipeline = XGBoostLearner("XGBClassifier"; max_depth=3, n_estimators=100)

# Create pipeline
pipeline = @pipeline imputer |> xgb_pipeline

# Fit and transform
fit!(pipeline, train_X, train_y)
predictions_pipeline = transform!(pipeline, test_X)

accuracy_pipeline = sum(predictions_pipeline .== test_y) / length(test_y)
println("Pipeline Accuracy: $(round(accuracy_pipeline * 100, digits=2))%")

# Example 7: List Available XGBoost Learners
println("\n7. Available XGBoost Learners")
println("-" ^ 70)
xgboostlearners()

println("\n" * "=" ^ 70)
println("Examples completed successfully!")
println("=" ^ 70)

# Additional Tips
println("\nTips for using XGBoost with AutoMLPipeline:")
println("  • Use XGBClassifier for classification tasks")
println("  • Use XGBRegressor for regression tasks")
println("  • Use XGBRFClassifier/XGBRFRegressor for random forest variants")
println("  • Tune hyperparameters like max_depth, learning_rate, n_estimators")
println("  • Consider using subsample and colsample_bytree for regularization")
println("  • XGBoost handles missing values automatically")
println("  • For large datasets, consider using tree_method='hist' or 'gpu_hist'")

# Made with Bob
