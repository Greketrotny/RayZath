#include "rzexception.hpp"

namespace RayZath
{
	// ~~~~~~~~ [STRUCT] Exception ~~~~~~~~
	Exception::Exception(
		const std::string& what,
		const std::string& file,
		const uint32_t& line)
		: what(what)
		, file(file)
		, line(line)
	{}

	std::string Exception::ToString() const noexcept
	{
		std::string str;
		str += "File: " + file + '\n';
		str += "Line: " + std::to_string(line) + '\n';
		str += "Exception: " + what + '\n';
		return str;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	// ~~~~~~~~ [STRUCT] CudaException ~~~~~~~~
	CudaException::CudaException(
		const std::string& what,
		const std::string& file,
		const uint32_t& line,
		const std::string& code_name,
		const uint32_t& code,
		const std::string& desc)
		: Exception(what, file, line)
		, code_name(code_name)
		, code(code)
		, desc(desc)
	{}
	CudaException::CudaException(
		const cudaError_t& cuda_error,
		const std::string& file,
		const uint32_t& line)
		: Exception("CUDA API exception", file, line)
		, code_name(cudaGetErrorName(cuda_error))
		, code(cuda_error)
		, desc(cudaGetErrorString(cuda_error))
	{}

	std::string CudaException::ToString() const noexcept
	{
		std::string str = Exception::ToString();
		str += "CUDA error: " + code_name + " (code: " + std::to_string(code) + ")\n";
		str += "Description: " + desc + '\n';
		return str;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}