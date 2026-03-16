module TestXGBoostLearners

using Test
using DataFrames
using AutoMLPipeline
using AutoMLPipeline.XGBoostLearners

function test_xgboost_classifier()
  @testset "XGBoost Classifier" begin
    # Load iris dataset
    iris = getiris()
    X = iris[:, 1:4]
    y = iris[:, 5] |> Vector

    # Split data
    (train_X, train_y), (test_X, test_y) = train_test_split(X, y, 0.7)

    # Test basic constructor
    xgb = XGBoostLearner("XGBClassifier")
    @test xgb isa XGBoostLearner
    @test xgb.model[:learner] == "XGBClassifier"

    # Test with parameters
    xgb_params = XGBoostLearner("XGBClassifier";
      max_depth=3,
      learning_rate=0.1,
      n_estimators=100)
    @test xgb_params.model[:impl_args][:max_depth] == 3
    @test xgb_params.model[:impl_args][:learning_rate] == 0.1
    @test xgb_params.model[:impl_args][:n_estimators] == 100

    # Test fit and transform
    fit!(xgb, train_X, train_y)
    predictions = transform!(xgb, test_X)

    @test length(predictions) == length(test_y)
    @test eltype(predictions) == String

    # Test accuracy
    accuracy = sum(predictions .== test_y) / length(test_y)
    @test accuracy > 0.8  # Should have reasonable accuracy on iris

    println("XGBoost Classifier test passed with accuracy: $(round(accuracy, digits=3))")
  end
end

function test_xgboost_regressor()
  @testset "XGBoost Regressor" begin
    # Create synthetic regression data
    n = 100
    X = DataFrame(
      x1=randn(n),
      x2=randn(n),
      x3=randn(n)
    )
    y = 2.0 .* X.x1 .+ 3.0 .* X.x2 .- 1.5 .* X.x3 .+ randn(n) .* 0.1

    # Split data
    (train_X, train_y), (test_X, test_y) = train_test_split(X, y, 0.7)

    # Test basic constructor
    xgb = XGBoostLearner("XGBRegressor")
    @test xgb isa XGBoostLearner
    @test xgb.model[:learner] == "XGBRegressor"

    # Test with parameters
    xgb_params = XGBoostLearner("XGBRegressor";
      max_depth=5,
      learning_rate=0.05,
      n_estimators=200)

    # Test fit and transform
    fit!(xgb_params, train_X, train_y)
    predictions = transform!(xgb_params, test_X)

    @test length(predictions) == length(test_y)
    @test eltype(predictions) == Float64

    # Test R² score (coefficient of determination)
    ss_res = sum((test_y .- predictions) .^ 2)
    ss_tot = sum((test_y .- mean(test_y)) .^ 2)
    r2 = 1 - ss_res / ss_tot
    @test r2 > 0.8  # Should have good R² on synthetic data

    println("XGBoost Regressor test passed with R²: $(round(r2, digits=3))")
  end
end

function test_xgboost_rf_classifier()
  @testset "XGBoost Random Forest Classifier" begin
    # Load iris dataset
    iris = getiris()
    X = iris[:, 1:4]
    y = iris[:, 5] |> Vector

    # Split data
    (train_X, train_y), (test_X, test_y) = train_test_split(X, y, 0.7)

    # Test XGBRFClassifier
    xgb_rf = XGBoostLearner("XGBRFClassifier";
      max_depth=3,
      n_estimators=50)

    fit!(xgb_rf, train_X, train_y)
    predictions = transform!(xgb_rf, test_X)

    @test length(predictions) == length(test_y)
    accuracy = sum(predictions .== test_y) / length(test_y)
    @test accuracy > 0.7

    println("XGBoost RF Classifier test passed with accuracy: $(round(accuracy, digits=3))")
  end
end

function test_xgboostlearners_list()
  @testset "XGBoost Learners List" begin
    # Test that xgboostlearners() function works
    println("\nAvailable XGBoost learners:")
    xgboostlearners()

    # Test that all expected learners are available
    expected_learners = ["XGBClassifier", "XGBRegressor", "XGBRFClassifier", "XGBRFRegressor", "XGBRanker"]
    for learner in expected_learners
      @test learner in keys(XGBoostLearners.learner_dict)
    end
  end
end

function test_xgboostoperator()
  @testset "XGBoost Operator" begin
    # Test xgboostoperator function
    xgb = xgboostoperator("XGBClassifier"; max_depth=3)
    @test xgb isa XGBoostLearner
    @test xgb.model[:learner] == "XGBClassifier"

    # Test error handling
    @test_throws ArgumentError xgboostoperator("NonExistentLearner")
  end
end

function run_tests()
  println("Running XGBoost Learners Tests...")
  println("="^50)

  test_xgboostlearners_list()
  test_xgboost_classifier()
  test_xgboost_regressor()
  test_xgboost_rf_classifier()
  test_xgboostoperator()

  println("\n" * "="^50)
  println("All XGBoost tests completed successfully!")
end

end # module

# Run tests if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
  using .TestXGBoostLearners
  TestXGBoostLearners.run_tests()
end

# Made with Bob
