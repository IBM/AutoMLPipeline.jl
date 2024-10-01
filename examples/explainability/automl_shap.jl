using AutoMLPipeline
using ShapML
using RDatasets
using DataFrames
using MLJ  # Machine learning
using Gadfly  # Plotting


# Load data.
boston = RDatasets.dataset("MASS", "Boston")
#------------------------------------------------------------------------------
# Train a machine learning model; currently limited to single outcome regression and binary classification.
outcome_name = "MedV"

# Data prep.
y, X = MLJ.unpack(boston, ==(Symbol(outcome_name)), colname -> true)

# load AutoMLPipeline operators
numf = NumFeatureSelector()
pca = skoperator("PCA")
ica = skoperator("FastICA")
fa  = skoperator("FactorAnalysis")
rb   = skoperator("RobustScaler")
pt   = skoperator("PowerTransformer")
norm = skoperator("Normalizer")
mx   = skoperator("MinMaxScaler")
std  = skoperator("StandardScaler")


# ShapML setup.
explain = copy(boston[1:300, :]) # Compute Shapley feature-level predictions for 300 instances.
explain = select(explain, Not(Symbol(outcome_name)))  # Remove the outcome column.
reference = copy(boston)  # An optional reference population to compute the baseline prediction.
reference = select(reference, Not(Symbol(outcome_name)))
sample_size = 30  # Number of Monte Carlo samples.

function amlp_predict_function(model, data)
  data_pred = DataFrame(y_pred = AutoMLPipeline.transform!(model, data))
  return data_pred
end

# model pipeline
pdec = (numf |> rb |> pca) + (numf |> std |> ica) + (numf |> mx |> fa)
skrf_reg = skoperator("PassiveAggressiveRegressor")
skrf_reg = skoperator("SGDRegressor")
skrf_reg = skoperator("AdaBoostRegressor")
skrf_reg = skoperator("RandomForestRegressor")
skrf_reg = skoperator("GradienBoostingRegressor")
skrf_reg = skoperator("SVR")
amlp_model = pdec |> skrf_reg
AutoMLPipeline.fit!(amlp_model,X,y)
data_shap = ShapML.shap(explain = explain,
                        reference = reference,
                        model = amlp_model,
                        predict_function = amlp_predict_function,
                        sample_size = sample_size,
                        seed = 1
                        )
show(data_shap, allcols = true)
g =  groupby(data_shap, :feature_name)
data_plot = combine(g, :shap_effect => mean)
data_plot = sort(data_plot, order(:shap_effect_mean, rev = true))
baseline = round(data_shap.intercept[1], digits = 1)
p1 = plot(data_plot, y = :feature_name, x = :shap_effect_mean, Coord.cartesian(yflip = true),
         Scale.y_discrete, Geom.bar(position = :dodge, orientation = :horizontal),
         Theme(bar_spacing = 1mm),
         Guide.xlabel("|Shapley effect| (baseline = $baseline)"), Guide.ylabel(nothing),
         Guide.title("Feature Importance - Mean Absolute Shapley Value"));
p1
data_plot = data_shap[data_shap.feature_name .== "Rm", :]  # Selecting 1 feature for ease of plotting.
baseline = round(data_shap.intercept[1], digits = 1)
p_points = layer(data_plot, x = :feature_value, y = :shap_effect, Geom.point())
p_line = layer(data_plot, x = :feature_value, y = :shap_effect, Geom.smooth(method = :loess, smoothing = 0.5),
               style(line_width = 0.75mm,), Theme(default_color = "black"))
p2 = plot(p_line, p_points, Guide.xlabel("Feature value"), Guide.ylabel("Shapley effect (baseline = $baseline)"),
         Guide.title("Feature Effect - $(data_plot.feature_name[1])"));
p2
