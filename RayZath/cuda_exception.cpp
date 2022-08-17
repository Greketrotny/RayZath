#include "cuda_exception.hpp"

#include "cuda_runtime_api.h"


namespace RayZath::Cuda
{
	Exception::Exception(const char* message)
		: RayZath::Exception(message)
	{}
	Exception::Exception(const std::string& message)
		: Exception(message.c_str())
	{}
	Exception::Exception(const char* message, const char* file, const int32_t line)
		: RayZath::Exception(file + std::to_string(line) + ": " + message)
	{}
	Exception::Exception(const std::string & message, const char* file, const int32_t line)
		: Exception(message.c_str(), file, line)
	{}

	void Exception::checkError(const cudaError_t cuda_error)
	{
		if (cuda_error != cudaSuccess)
		{
			using namespace std::string_literals;
			RZThrowCoreCUDA(
				"Internal CUDA error:\n"s +
				cudaGetErrorName(cuda_error) + " (" + std::to_string(cuda_error) + ")\n" +
				cudaGetErrorString(cuda_error));
		}
	}
	void Exception::checkError(const cudaError_t cuda_error, const char* file, const uint32_t line)
	{
		if (cuda_error != cudaSuccess)
		{
			using namespace std::string_literals;
			RZThrowCoreCUDA(
				"Internal CUDA error: ("s + file + ':' + std::to_string(line) + ")\n" +
				cudaGetErrorName(cuda_error) + " (" + std::to_string(cuda_error) + ")\n" +
				cudaGetErrorString(cuda_error));
		}
	}
}