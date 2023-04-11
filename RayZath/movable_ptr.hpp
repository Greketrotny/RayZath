#ifndef RZ_MOVABLE_PTR_HPP
#define RZ_MOVABLE_PTR_HPP

#include <utility>

namespace RayZath
{
	template <typename T>
	class MovablePointer
	{
	private:
		T* m_ptr = nullptr;

	public:
		MovablePointer() noexcept = default;
		MovablePointer(std::nullptr_t) noexcept {};
		MovablePointer(const MovablePointer& other) noexcept
			: m_ptr(other.m_ptr)
		{}
		MovablePointer(MovablePointer&& other) noexcept
			: m_ptr(std::exchange(other.m_ptr, nullptr))
		{}
		MovablePointer(T* ptr) noexcept
			: m_ptr(ptr)
		{}

		MovablePointer& operator=(const MovablePointer& other) noexcept
		{
			m_ptr = other.m_ptr;
			return *this;
		}
		MovablePointer& operator=(MovablePointer&& other) noexcept
		{
			if (this != &other) m_ptr = std::exchange(other.m_ptr, nullptr);
			return *this;
		}
		MovablePointer& operator=(T* ptr) noexcept
		{
			m_ptr = ptr;
			return *this;
		}
		T* operator->() const noexcept
		{
			return m_ptr;
		}
		T& operator*() const noexcept
		{
			return *m_ptr;
		}

		explicit operator bool() const noexcept
		{
			return m_ptr != nullptr;
		}

		T* get() const noexcept
		{
			return m_ptr;
		}
	};
}

#endif
