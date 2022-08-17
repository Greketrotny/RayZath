#ifndef CUDA_EXCEPTION_HPP
#define CUDA_EXCEPTION_HPP

#include "rzexception.hpp"
#include "driver_types.h"

namespace RayZath::Cuda
{
	struct Exception : public RayZath::Exception
	{
	public:
		Exception(const char* message);
		Exception(const std::string& message);
		Exception(const char* message, const char* file, const int32_t line);
		Exception(const std::string& message, const char* file, const int32_t line);

		static void checkError(const cudaError_t cuda_error);
		static void checkError(const cudaError_t cuda_error, const char* file, const uint32_t line);
	};

	#define RZThrowCoreCUDA(message)\
	{\
		throw RayZath::Cuda::Exception(message);\
	}

	#if (defined(DEBUG) || defined(_DEBUG))
	#define RZAssertCoreCUDA(cuda_error)\
	{\
		RayZath::Cuda::Exception::checkError(cuda_error, __FILE__, __LINE__);\
	}
	#else
	#define RZAssertCoreCUDA(cuda_error)\
	{\
		RayZath::Cuda::Exception::checkError(cuda_error);\
	}
	#endif
}

#endif
