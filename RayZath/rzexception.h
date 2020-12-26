#ifndef RZEXCEPTION_H
#define RZEXCEPTION_H

#include "cuda_include.h"
#include <string>

namespace RayZath
{
	struct Exception
	{
	public:
		const std::wstring file;
		const uint32_t line;
		const std::wstring what;


	public:
		Exception(
			const std::wstring& file = L"NA",
			const uint32_t& line = 0u,
			const std::wstring& what = L"Unknown")
			: file(file)
			, line(line)
			, what(what)
		{}
		Exception(
			const std::string& file = "NA",
			const uint32_t& line = 0u,
			const std::wstring& what = L"Unknown")
			: file(file.begin(), file.end())
			, line(line)
			, what(what)
		{}
		~Exception() {}


		std::wstring ToString() const noexcept
		{
			std::wstring str = L"";
			str += L"File: " + file + L"\n";
			str += L"Line: " + std::to_wstring(line) + L"\n";
			str += L"Exception: " + what + L"\n";
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
		const std::wstring code_name;
		const uint32_t code;
		const std::wstring desc;


	public:
		CudaException(
			const std::wstring& file = L"NA",
			const uint32_t& line = 0u,
			const std::wstring& what = L"Unknown",
			const std::wstring& code_name = L"Unknown",
			const uint32_t& code = 0u,
			const std::wstring& desc = L"NA")
			: Exception(file, line, what)
			, code_name(code_name)
			, code(code)
			, desc(desc)
		{}


		std::wstring ToString() const noexcept
		{
			std::wstring str = Exception::ToString();
			str += L"CUDA error: " + code_name + L" (code: " + std::to_wstring(code) + L")\n";
			str += L"Description: " + desc + L"\n";
			return str;
		}

		static void CheckCudaError(const char* file, const uint32_t line, cudaError_t cuda_error)
		{
			if (cuda_error != cudaSuccess)
			{
				std::string sfile(file);
				std::wstring wfile(sfile.begin(), sfile.end());

				std::string sname(cudaGetErrorName(cuda_error));
				std::wstring wname(sname.begin(), sname.end());

				std::string sdesc(cudaGetErrorString(cuda_error));
				std::wstring wdesc(sdesc.begin(), sdesc.end());

				throw CudaException(
					wfile, line, L"CUDA API exception",
					wname, cuda_error, wdesc);
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