#ifndef RZEXCEPTION_H
#define RZEXCEPTION_H

#include "driver_types.h"

#include <string>
#include <stdexcept>

namespace RayZath
{
	struct Exception : public std::runtime_error
	{
	public:
		Exception(const char* message);
		Exception(const std::string& message);
		Exception(const Exception& other, const char* message);
		Exception(const Exception& other, const std::string& message);
	};
	struct CoreException : public std::logic_error
	{
	public:
		CoreException(const char* message);
		CoreException(const std::string& message);
		CoreException(const char* message, const char* file, const int32_t line);
		CoreException(const std::string& message, const char* file, const int32_t line);
	};


	#define	RZAssert(condition, message)\
	{\
		const auto& _condition = condition;\
		const auto& _message = message;\
		if (!bool(_condition)) throw RayZath::Exception(_message);\
	}
	#define RZThrow(message)\
	{\
		const auto& _message = message;\
		throw RayZath::Exception(_message);\
	}

	#if (defined(DEBUG) || defined(_DEBUG))
	#define	RZAssertCore(condition, message)\
	{\
		const auto& _condition = condition;\
		const auto& _message = message;\
		if (!bool(_condition)) throw RayZath::CoreException(_message, __FILE__, __LINE__);\
	}
	#define RZThrowCore(message)\
	{\
		const auto& _message = message;\
		throw RayZath::CoreException(_message, __FILE__, __LINE__);\
	}
	#else
	#define	RZAssertCore(condition, message)\
	{\
		const auto& _condition = condition;\
		const auto& _message = message;\
		if (!bool(_condition)) throw RayZath::CoreException(_message);\
	}
	#define RZThrowCore(message)\
	{\
		const auto& _message = message;\
		throw RayZath::CoreException(_message);\
	}
	#endif
}

#endif