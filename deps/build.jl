using PyCall: pyimport_conda, pycall
using Conda

function installpypackage()
	try
		pyimport_conda("sklearn", "scikit-learn")
      println("scikit-learn successfully installed")
	catch
		try
			Conda.add("scikit-learn")
			pyimport_conda("sklearn", "scikit-learn")
			pyimport_conda("sklearn.decomposition", "scikit-learn")
			println("scikit-learn successfully installed")
		catch
			println("scikit-learn failed to install")
		end
	end
end

installpypackage()
