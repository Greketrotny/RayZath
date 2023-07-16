#ifndef RZ_SHARED_RESOURCE_HPP
#define RZ_SHARED_RESOURCE_HPP

#include "rzexception.hpp"
#include "movable_ptr.hpp"

#include <cstdint>
#include <cstddef>
#include <assert.h>
#include <shared_mutex>
#include <atomic>
#include <memory>

namespace RayZath::Engine
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
		explicit operator bool() const noexcept
		{
			return mp_object != nullptr;
		}

		T* get() noexcept
		{
			return mp_object;
		}
		const T* get() const noexcept
		{
			return mp_object;
		}
		uint32_t id() const noexcept
		{
			RZAssertCore(mp_object != nullptr, "Invalid Accessor.");
			return m_id;
		}
		std::shared_mutex& mtx() const
		{
			return m_mtx;
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
		using accessor_t = Accessor<std::remove_const_t<T>>;

	private:
		std::shared_ptr<accessor_t> m_accessor{};
		lock_t m_lck{};
		MovablePointer<T> mp_object{};

	public:
		Ref(std::shared_ptr<accessor_t> accessor)
			: m_accessor(std::move(accessor))
		{
			if (!m_accessor) 
				return;
			m_lck = lock_t(m_accessor->mtx());
			mp_object = m_accessor->get();
		}
		Ref(Ref&& other) noexcept = default;
		Ref(const Ref& other) = delete;

		template <class rhsT>
		bool operator==(const Ref<rhsT>& other) const noexcept
		{
			return m_accessor == other.m_accessor;
		}
		template <class rhsT>
		bool operator<(const Ref<rhsT>& other) const noexcept
		{
			return m_accessor <= other.m_accessor;
		}
		explicit operator bool() const noexcept
		{
			return m_accessor ? bool(*m_accessor) : false;
		}
		T* operator->() const noexcept
		{
			return mp_object;
		}
		T& operator*() const noexcept 
		{
			return *mp_object;
		}

		uint32_t id() const noexcept
		{
			return m_accessor->id();
		}
	};

	template <class T>
	class BRef
	{
	public:
		using lock_t = std::conditional_t<
			std::is_const_v<T>,
			std::shared_lock<std::shared_mutex>, std::unique_lock<std::shared_mutex>>;

	private:
		lock_t m_lck;
		std::reference_wrapper<T> m_object;

	public:
		BRef(std::reference_wrapper<T> object)
			: m_lck(object.get().mtx())
			, m_object(std::move(object))
		{}
		BRef(BRef&& other) noexcept = default;
		BRef(const BRef& other) = delete;
		
		T* operator->() const noexcept
		{
			return &m_object.get();
		}
		T& operator*() const noexcept
		{
			return m_object;
		}
	};


	template <class T>
	class Handle
	{
	public:
		using accessor_t = Accessor<std::remove_const_t<T>>;

	private:
		std::shared_ptr<accessor_t> m_accessor;

	public:
		Handle() = default;
		Handle(const Handle& other) = default;
		Handle(Handle&& other) noexcept = default;
		Handle(std::shared_ptr<accessor_t> accessor)
			: m_accessor(std::move(accessor))
		{}

		Handle& operator=(const Handle& other) = default;
		Handle& operator=(Handle&& other) noexcept = default;
		bool operator==(const Handle& other) const noexcept
		{
			return m_accessor == other.m_accessor;
		}
		bool operator==(const T& object) const noexcept
		{
			return bool(m_accessor) ? m_accessor->get() == &object : false;
		}
		bool operator<(const Handle& other) const noexcept
		{
			return m_accessor < other.m_accessor;
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
			return Ref<T>(m_accessor);
		}
		auto cref()
		{
			return Ref<const T>(m_accessor);
		}
	};

	template <class T>
	class Owner
	{
	public:
		using accessor_t = Accessor<std::remove_const_t<T>>;

	private:
		std::weak_ptr<accessor_t> m_accessor;

	public:
		Owner() = default;
		Owner(std::weak_ptr<accessor_t> accessor)
			: m_accessor(std::move(accessor))
		{}
		Owner(const Owner& other) = delete;
		Owner(Owner&& other) noexcept = default;
		Owner(Owner&& other, T* new_object_address, uint32_t new_id) noexcept
			: m_accessor(std::move(other.m_accessor))
		{
			if (auto accessor = m_accessor.lock(); accessor) 
				accessor->setLocation(new_object_address, new_id);
		}
		~Owner()
		{
			if (auto accessor = m_accessor.lock(); accessor)
				accessor->reset();
		}

		Owner& operator=(const Owner& other) = delete;
		Owner& operator=(Owner&& other) noexcept = default;
		explicit operator bool() const noexcept
		{
			return !m_accessor.expired();
		}

		void accessor(std::weak_ptr<accessor_t> accessor)
		{
			m_accessor = std::move(accessor);
		}
		Handle<T> handle()
		{
			return Handle<T>(m_accessor.lock());
		}
	};
}

namespace std
{
	template <typename T>
	struct hash<RayZath::Engine::Handle<T>>
	{
		std::size_t operator()(const RayZath::Engine::Handle<T>& handle) const
		{
			return handle.hash();
		}
	};
	template <typename T>
	struct less<RayZath::Engine::Handle<T>>
	{
		bool operator()(const RayZath::Engine::Handle<T>& left, const RayZath::Engine::Handle<T>& right) const
		{
			return left < right;
		}
	};
}

#endif
