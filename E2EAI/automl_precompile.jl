import AMLPipelineBase
include(joinpath(pkgdir(AMLPipelineBase), "test", "runtests.jl"))

import AutoMLPipeline
include(joinpath(pkgdir(AutoMLPipeline), "test", "runtests.jl"))
