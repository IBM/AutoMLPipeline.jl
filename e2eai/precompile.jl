import AMLPipelineBase
include(joinpath(pkgdir(AMLPipelineBase), "test", "runtests.jl"))

import TSML
include(joinpath(pkgdir(TSML), "test", "runtests.jl"))
