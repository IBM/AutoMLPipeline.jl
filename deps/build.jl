using PyCall: pyimport_conda, pycall
using Pkg
using Conda

function installpypackage()
	try
		pyimport_conda("mkl", "mkl")
		pyimport_conda("sklearn", "scikit-learn")
      println("mkl and scikit-learn successfully installed")
	catch
		try
         #https://github.com/JuliaPy/Conda.jl/issues/182
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

installpypackage()
