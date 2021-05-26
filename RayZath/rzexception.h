#ifndef RZEXCEPTION_H
#define RZEXCEPTION_H

#include "cuda_include.h"
#include <string>

namespace RayZath
{
	struct Exception
	{
	public:
		const std::string file;
		const uint32_t line;
		const std::string what;


	public:
		Exception(
			const std::string& file = "unknown",
			const uint32_t& line = 0u,
			const std::string& what = "unknown")
			: file(file)
			, line(line)
			, what(what)
		{}

		std::string ToString() const noexcept
		{
			std::string str;
			str += "File: " + file + '\n';
			str += "Line: " + std::to_string(line) + '\n';
			str += "Exception: " + what + '\n';
			return str;
		}
	};

	#if (defined(DEBUG) || defined(_DEBUG))
	#define ThrowException(what) throw Exception(__FILE__, __LINE__, (what))
	#define RZAssert(cond, what) if (!bool(cond)) throw Exception(__FILE__, __LINE__, (what));
	#else
	#define ThrowException(what)
	#define RZAssert(cond, what)
	#endif

	struct CudaException : public Exception
	{
	public:
		const std::string code_name;
		const uint32_t code;
		const std::string desc;


	public:
		CudaException(
			const std::string& file = "unknown",
			const uint32_t& line = 0u,
			const std::string& what = "unknown",
			const std::string& code_name = "unknown",
			const uint32_t& code = 0u,
			const std::string& desc = "unknown")
			: Exception(file, line, what)
			, code_name(code_name)
			, code(code)
			, desc(desc)
		{}


		std::string ToString() const noexcept
		{
			std::string str = Exception::ToString();
			str += "CUDA error: " + code_name + " (code: " + std::to_string(code) + ")\n";
			str += "Description: " + desc + '\n';
			return str;
		}

		static void CheckCudaError(const char* file, const uint32_t line, cudaError_t cuda_error)
		{
			if (cuda_error != cudaSuccess)
			{
				std::string sfile(file);
				std::wstring wfile(sfile.begin(), sfile.end());

				std::string name(cudaGetErrorName(cuda_error));
				std::string desc(cudaGetErrorString(cuda_error));

				throw CudaException(
					file, line, "CUDA API exception",
					name, cuda_error, desc);
			}
		}
	};

	#if (defined(DEBUG) || defined(_DEBUG))
	#define CudaErrorCheck(cuda_error) { RayZath::CudaException::CheckCudaError(__FILE__, __LINE__, (cuda_error)); }	
	#else
	#define CudaErrorCheck(cuda_error) {(cuda_error);}
	#endif
}

#endif