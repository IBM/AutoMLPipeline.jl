using PyCall: pyimport_conda, pycall
using Conda

function installmac()
   try
      pyimport_conda("nomkl", "nomkl")
      pyimport_conda("sklearn", "scikit-learn")
      println("mkl and scikit-learn successfully installed")
   catch
      try
         # https://github.com/JuliaPy/Conda.jl/issues/182
         # https://docs.anaconda.com/mkl-optimizations/
         Conda.add("nomkl")
         Conda.add("scikit-learn")
         Conda.rm("mkl")
         pyimport_conda("sklearn", "scikit-learn")
         pyimport_conda("sklearn.decomposition", "scikit-learn")
         pyimport_conda("mkl", "mkl")
         println("mkl and scikit-learn successfully installed")
      catch
         println("mkl and scikit-learn failed to install")
      end
   end
end

function installwinlinux()
   try
      pyimport_conda("sklearn", "scikit-learn")
      println("scikit-learn successfully installed")
   catch
      try
         # https://github.com/JuliaPy/Conda.jl/issues/182
         # https://docs.anaconda.com/mkl-optimizations/
         Conda.add("scikit-learn")
         pyimport_conda("sklearn", "scikit-learn")
         pyimport_conda("sklearn.decomposition", "scikit-learn")
         println("scikit-learn successfully installed")
      catch
         println("scikit-learn failed to install")
      end
   end
end

if Sys.isapple()
   installmac()
else
   installwinlinux()
end
