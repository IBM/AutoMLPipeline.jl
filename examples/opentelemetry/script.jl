using AutoMLPipeline
using OpenTelemetry

# enable tracing with docker
# docker-compose up

enableotlp()

with_span("crossvalidation") do
    include(joinpath(
        pkgdir(AutoMLPipeline),
        "test/test_skcrossvalidator.jl"))
end


with_span("learners") do
    include(joinpath(
        pkgdir(AutoMLPipeline),
        "test/test_sklearner.jl"))
end


with_span("preprocessor") do
    include(joinpath(
        pkgdir(AutoMLPipeline),
        "test/test_skpreprocessing.jl"))
end
