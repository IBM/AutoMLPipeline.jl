using PyCall: pyimport_conda, pycall
using Conda

function installpypackage()
	try
		pyimport_conda("sklearn", "scikit-learn")
	catch
		try
			Conda.add("scikit-learn")
		catch
			println("scikit-learn failed to install")
		end
	end
end

installpypackage()
