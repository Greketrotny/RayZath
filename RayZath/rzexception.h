#ifndef RZEXCEPTION_H
#define RZEXCEPTION_H

#include "cuda_include.h"
#include <string>

namespace RayZath
{
	struct Exception
	{
	public:
		std::string what;
		std::string file;
		uint32_t line;


	public:
		Exception(
			const std::string& what = "unknown",
			const std::string& file = "unknown",
			const uint32_t& line = 0u);

		std::string ToString() const noexcept;
	};

	struct CudaException : public Exception
	{
	public:
		std::string code_name;
		uint32_t code;
		std::string desc;


	public:
		CudaException(
			const std::string& what = "unknown",
			const std::string& file = "unknown",
			const uint32_t& line = 0u,
			const std::string& code_name = "unknown",
			const uint32_t& code = 0u,
			const std::string& desc = "unknown");
		CudaException(
			const cudaError_t& cuda_error,
			const std::string& file,
			const uint32_t& line);

		std::string ToString() const noexcept;
		static void CheckCudaError(cudaError_t cuda_error, const char* file, const uint32_t line)
		{
			if (cuda_error != cudaSuccess)
			{
				throw CudaException(
					cuda_error, file, line);
			}
		}
	};

	#define RZThrow(what) throw RayZath::Exception((what), __FILE__, __LINE__);
	#define RZAssert(cond, what) if (!bool(cond)) throw RayZath::Exception((what), __FILE__, __LINE__);

	#if (defined(DEBUG) || defined(_DEBUG))
	#define RZThrowDebug(what) RZThrow(what)
	#define RZAssertDebug(cond, what) RZAssert(cond, what)

	#define ThrowException(what) throw RayZath::Exception((what), __FILE__, __LINE__)
	#define CudaErrorCheck(cuda_error) { RayZath::CudaException::CheckCudaError((cuda_error), __FILE__, __LINE__); }
	#define ThrowCudaException(cuda_error) { throw RayZath::CudaException((cuda_error), __FILE__, __LINE__ ); }
	#else
	#define RZThrowDebug(what) ;
	#define RZAssertDebug(cond, what) ;

	#define CudaErrorCheck(cuda_error) {(cuda_error);}
	#define ThrowCudaException(cuda_error) {(cuda_error);}
	#define ThrowException(what) ;
	#endif
}

#endif