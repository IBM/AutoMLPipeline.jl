using PyCall: pyimport_conda, pycall
using Conda

function installpypackage()
	try
		pyimport_conda("sklearn", "scikit-learn")
		pyimport_conda("mkl", "mkl")
      println("mkl and scikit-learn successfully installed")
	catch
		try
			Conda.add("mkl")
			Conda.add("scikit-learn")
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
