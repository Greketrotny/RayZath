#ifndef RZ_SHARED_RESOURCE_HPP
#define RZ_SHARED_RESOURCE_HPP

#include "rzexception.hpp"

#include <cstdint>
#include <cstddef>
#include <assert.h>
#include <shared_mutex>
#include <atomic>
#include <memory>

namespace RayZath::Engine::SR
{
	template <class T>
	class Handle;
	template <class T>
	class Owner;

	template <class T>
	class Accessor
	{
	private:
		using object_t = T;

		T* mp_object;
		uint32_t m_id;
		std::shared_mutex& m_mtx;

	public:
		Accessor(T* object, const uint32_t id, std::shared_mutex& mtx) noexcept
			: mp_object(object)
			, m_id(id)
			, m_mtx(mtx)
		{}
		Accessor(const Accessor& other) = delete;
		Accessor(Accessor&& other) = delete;

		Accessor& operator=(const Accessor& other) = delete;
		Accessor& operator=(Accessor&& other) = delete;

		std::shared_mutex& mtx() const
		{
			return m_mtx;
		}
		uint32_t id() const noexcept
		{
			RZAssertCore(mp_object != nullptr, "Invalid Handle.");
			return m_id;
		}
		void setLocation(T* new_p_object, uint32_t new_id)
		{
			mp_object = new_p_object;
			m_id = new_id;
		}
		void reset()
		{
			mp_object = nullptr;
		}
	};

	template <class T>
	class Ref
	{
	public:
		using lock_t = std::conditional_t<
			std::is_const_v<T>,
			std::shared_lock<std::shared_mutex>, std::unique_lock<std::shared_mutex>>;
	private:
		std::shared_ptr<Accessor<T>> m_accessor;
		lock_t m_lck;
		std::reference_wrapper<T> m_object;

	public:
		Ref(std::shared_ptr<Accessor<T>> accessor)
			: m_accessor(std::move(accessor))
			, m_lck(m_accessor.get().mtx())
			, m_object(m_accessor.get())
		{}
		Ref(Ref&& other) noexcept = default;
		Ref(const Ref& other) = delete;

		template <class rhsT>
		bool operator==(const Ref<rhsT>& other)
		{
			return m_accessor == other.m_accessor;
		}
		template <class rhsT>
		bool operator<(const Ref<rhsT>& other)
		{
			return m_accessor <= other.m_accessor;
		}
		T* operator->() const
		{
			return &m_object.get();
		}
		T& operator*() const
		{
			return m_object.get();
		}

		uint32_t id() const noexcept
		{
			return m_accessor->id();
		}
	};


	template <class T>
	class Handle
	{
	private:
		std::shared_ptr<Accessor<T>> m_accessor;

	public:
		Handle() = default;
		Handle(const Handle& other) = default;
		Handle(Handle&& other) noexcept = default;
		Handle(std::shared_ptr<Accessor<T>> accessor)
			: m_accessor(std::move(accessor))
		{}

		Handle& operator=(const Handle& other) = default;
		Handle& operator=(Handle&& other) noexcept = default;
		bool operator==(const Handle& other) const
		{
			return m_accessor == other.m_accessor;
		}
		bool operator<(const Handle& other) const noexcept
		{
			return m_accessor < other.m_accessor;
		}

		explicit operator bool() const noexcept
		{
			return m_accessor ? bool(*m_accessor) : false;
		}

		void reset()
		{
			m_accessor.reset();
		}
		std::size_t hash() const noexcept
		{
			return static_cast<std::size_t>(m_accessor.get());
		}
		auto ref()
		{
			RZAssertCore(m_accessor, "Attempt to create a ref from an invalid Handle.");
			return Ref<T>(m_accessor);
		}
		auto cref()
		{
			RZAssertCore(m_accessor, "Attempt to create a cref from an invalid Handle.");
			return Ref<const T>(m_accessor);
		}

		friend class Owner<T>;
	};

	template <class T>
	class Owner
	{
	private:
		std::shared_ptr<Accessor<T>> m_accessor;

	public:
		Owner() = default;
		Owner(T* object, uint32_t id)
			: m_accessor(new Accessor<T>(object, id))
		{}
		Owner(const Owner& other) = delete;
		Owner(Owner&& other) noexcept = default;
		Owner(Owner&& other, T* new_object_address, uint32_t new_id) noexcept
		{
			m_accessor = other.m_accessor;
			if (m_accessor) m_accessor->setLocation(new_object_address, new_id);
		}
		~Owner()
		{
			if (m_accessor)
				m_accessor->reset();
		}

		Owner& operator=(const Owner& other) = delete;
		Owner& operator=(Owner&& other) = delete;
		explicit operator bool() const noexcept
		{
			return bool(m_accessor);
		}

		void accessor(std::shared_ptr<Accessor<T>> accessor)
		{
			m_accessor = std::move(accessor);
		}
		Handle<T> handle()
		{
			return Handle<T>(m_accessor);
		}
	};
}

namespace std
{
	template <typename T>
	struct hash<RayZath::Engine::SR::Handle<T>>
	{
		std::size_t operator()(const RayZath::Engine::SR::Handle<T>& handle) const
		{
			return handle.hash();
		}
	};
	template <typename T>
	struct less<RayZath::Engine::SR::Handle<T>>
	{
		bool operator()(const RayZath::Engine::SR::Handle<T>& left, const RayZath::Engine::SR::Handle<T>& right) const
		{
			return left < right;
		}
	};
}

#endif
