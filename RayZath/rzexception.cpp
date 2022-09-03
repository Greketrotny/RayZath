#include "rzexception.hpp"

#include "cuda_include.hpp"

#include <stdexcept>

namespace RayZath
{
	using namespace std::string_literals;

	Exception::Exception(const char* message)
		: runtime_error(message)
	{}
	Exception::Exception(const std::string& message)
		: Exception(message.c_str())
	{}
	Exception::Exception(const Exception& other, const char* message)
		: runtime_error(std::string(message) + other.what())
	{}
	Exception::Exception(const Exception& other, const std::string& message)
		: Exception(other, message.c_str())
	{}

	CoreException::CoreException(const char* message)
		: logic_error(message)
	{}
	CoreException::CoreException(const std::string& message)
		: CoreException(message.c_str())
	{}
	CoreException::CoreException(const char* message, const char* file, const int32_t line)
		: logic_error(file + ":"s + std::to_string(line) + ": " + message)
	{}
	CoreException::CoreException(const std::string& message, const char* file, const int32_t line)
		: CoreException(message.c_str(), file, line)
	{}
}
