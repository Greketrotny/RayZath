#ifndef ROHO_H
#define ROHO_H

#include <assert.h>
#include <vector>
#include <functional>
#include <memory>

namespace RayZath::Engine
{
	template <class T>
	struct Handle;
	template <class T>
	struct Owner;
	template <class T>
	struct Observer;

	template <class T>
	struct Accessor
	{
	private:
		T* mp_object;
		uint32_t m_idx;
		size_t m_ref_count;
		std::vector<Observer<T>*> m_observers;


	public:
		Accessor(T* object, const uint32_t idx)
			: mp_object(object)
			, m_idx(idx)
			, m_ref_count(1u)
		{}
		Accessor(const Accessor& other) = delete;
		Accessor(Accessor&& other) = delete;
		~Accessor()
		{
			assert(m_ref_count == 0u);
			assert(m_observers.size() == 0u);

			destroy();
		}


	public:
		Accessor& operator=(const Accessor& other) = delete;
		Accessor& operator=(Accessor&& other) = delete;
		explicit operator bool() const noexcept
		{
			return mp_object != nullptr;
		}


	public:
		T* get() const
		{
			return mp_object;
		}

		uint32_t idx() const
		{
			return m_idx;
		}
		void idx(uint32_t idx)
		{
			m_idx = idx;
			notifyObservers();
		}

		size_t incRefCount()
		{
			return ++m_ref_count;
		}
		size_t decRefCount()
		{
			assert(m_ref_count > 0u);
			return --m_ref_count;
		}
		void subscribeObserver(Observer<T>* observer)
		{
			m_observers.push_back(observer);
		}
		void unsubscribeObserver(const Observer<T>* observer)
		{
			assert(m_observers.size() > 0u);

			auto iter = std::find(m_observers.begin(), m_observers.end(), observer);
			if (iter != m_observers.end())
			{
				m_observers.erase(iter);
			}
			else
				assert(false && "Unrecognized observer tried to unsubscribe!");
		}

		void destroy()
		{
			if (mp_object)
			{
				delete mp_object;
				mp_object = nullptr;
				notifyObservers();
			}
		}
	private:
		void notifyObservers();
	};


	template <class T>
	struct Handle
	{
	private:
		Accessor<T>* mp_accessor;


	public:
		Handle()
			: mp_accessor(nullptr)
		{}
		Handle(const Handle& other)
		{
			mp_accessor = other.mp_accessor;
			if (mp_accessor) mp_accessor->incRefCount();
		}
		Handle(Handle&& other) noexcept
		{
			mp_accessor = other.mp_accessor;
			other.mp_accessor = nullptr;
		}
		Handle(const Owner<T>& owner);
		Handle(const Observer<T>& observer);
		~Handle()
		{
			if (mp_accessor)
			{
				if (mp_accessor->decRefCount() == 0u)
					delete mp_accessor;
				mp_accessor = nullptr;
			}
		}

	public:
		Handle& operator=(const Handle& other)
		{
			if (this == &other) 
				return *this;
			if (mp_accessor == other.mp_accessor) 
				return *this;

			if (mp_accessor)
			{
				if (mp_accessor->decRefCount() == 0u)
					delete mp_accessor;
			}
			mp_accessor = other.mp_accessor;
			if (mp_accessor) mp_accessor->incRefCount();

			return *this;
		}
		Handle& operator=(Handle&& other) noexcept
		{
			if (this == &other) 
				return *this;
			if (mp_accessor == other.mp_accessor) 
				return *this;

			if (mp_accessor)
			{
				if (mp_accessor->decRefCount() == 0u)
					delete mp_accessor;
			}
			mp_accessor = other.mp_accessor;
			other.mp_accessor = nullptr;

			return *this;
		}
		Handle& operator=(const Owner<T>& owner);
		Handle& operator=(const Observer<T>& observer);
		bool operator==(const Handle& other) const
		{
			return (mp_accessor == other.mp_accessor);
		}
		bool operator==(const Owner<T>& owner) const;
		bool operator==(const Observer<T>& observer) const;
		T* operator->() const noexcept
		{
			return mp_accessor ? mp_accessor->get() : nullptr;
		}
		T& operator*() const
		{
			return *mp_accessor->get();
		}
		explicit operator bool() const noexcept
		{
			return mp_accessor ? bool(*mp_accessor) : false;
		}


	public:
		void release()
		{
			if (mp_accessor)
			{
				if (mp_accessor->decRefCount() == 0u)
					delete mp_accessor;
				mp_accessor = nullptr;
			}
		}
		const Accessor<T>* accessor() const
		{
			return mp_accessor;
		}

		friend struct Owner<T>;
		friend struct Observer<T>;
	};

	template <class T>
	struct Owner : public Handle<T>
	{
	private:
		using Handle<T>::mp_accessor;
		using Handle<T>::release;


	public:
		Owner() = default;
		Owner(T* object, const uint32_t idx)
		{
			mp_accessor = new Accessor<T>(object, idx);
		}
		Owner(const Owner& other) = delete;
		Owner(Owner&& other) noexcept
		{
			mp_accessor = other.mp_accessor;
			other.mp_accessor = nullptr;
		}
		~Owner()
		{
			if (mp_accessor) mp_accessor->destroy();
		}

	public:
		Owner& operator=(const Owner& other) = delete;
		Owner& operator=(Owner&& other) noexcept
		{
			if (this == &other)
				return *this;

			destroy();
			mp_accessor = other.mp_accessor;
			other.mp_accessor = nullptr;

			return *this;
		}

	public:
		void destroy()
		{
			if (mp_accessor) mp_accessor->destroy();
			release();
		}
		void reasign(T* object, uint32_t idx)
		{
			destroy();
			mp_accessor = new Accessor<T>(object, idx);
		}
		Accessor<T>* accessor()
		{
			return mp_accessor;
		}


		friend struct Handle<T>;
		friend struct Observer<T>;
	};

	template <class T>
	struct Observer : public Handle<T>
	{
	private:
		using Handle<T>::mp_accessor;
		std::function<void()> m_notify_function;


	public:
		Observer() = default;
		Observer(const std::function<void()>& f)
			: m_notify_function(f)
		{}
		Observer(const Observer& other)
			: Handle<T>(static_cast<const Handle<T>&>(other))
			, m_notify_function(other.m_notify_function)
		{
			if (mp_accessor) mp_accessor->subscribeObserver(this);
		}
		Observer(Observer&& other) noexcept 
		{
			// notify function
			m_notify_function = other.m_notify_function;
			other.m_notify_function = nullptr;

			// accessor
			if (other.mp_accessor)
			{
				other.mp_accessor->unsubscribeObserver(&other);
				other.mp_accessor->subscribeObserver(this);
			}
			mp_accessor = other.mp_accessor;
			other.mp_accessor = nullptr;
		}
		Observer(const Owner<T>& owner);
		Observer(const Handle<T>& handle)
			: Observer(handle, nullptr)
		{}
		Observer(const Handle<T>& handle, const std::function<void()>& f);
		~Observer()
		{
			if (mp_accessor) mp_accessor->unsubscribeObserver(this);
		}


	public:
		Observer& operator=(const Observer& other)
		{
			if (this == &other) return *this;
			if (mp_accessor == other.mp_accessor) return *this;

			// notify function
			m_notify_function = other.m_notify_function;

			// accessor
			if (mp_accessor)
			{
				mp_accessor->unsubscribeObserver(this);
				if (mp_accessor->decRefCount() == 0u) delete mp_accessor;
			}
			mp_accessor = other.mp_accessor;
			if (mp_accessor)
			{
				mp_accessor->incRefCount();
				mp_accessor->subscribeObserver(this);
			}

			return *this;
		}
		Observer& operator=(Observer&& other) noexcept
		{
			if (this == &other) return *this;
			if (mp_accessor == other.mp_accessor) return *this;

			// notify function
			m_notify_function = other.m_notify_function;
			other.m_notify_function = nullptr;

			// accessor
			if (mp_accessor)
			{
				mp_accessor->unsubscribeObserver(this);
				if (mp_accessor->decRefCount() == 0u) delete mp_accessor;
			}
			if (other.mp_accessor)
			{
				other.mp_accessor->unsubscribeObserver(&other);
				other.mp_accessor->subscribeObserver(this);
			}
			mp_accessor = other.mp_accessor;
			other.mp_accessor = nullptr;

			return *this;
		}
		Observer& operator=(const Owner<T>& owner);
		Observer& operator=(const Handle<T>& handle);


	public:
		void setNotifyFunction(const std::function<void()>& f)
		{
			m_notify_function = f;
		}
		const std::function<void()>& GetNotifyFunction()
		{
			return m_notify_function;
		}
		void release()
		{
			if (mp_accessor) mp_accessor->unsubscribeObserver(this);
			Handle<T>::release();
		}
	private:
		void notify()
		{
			if (m_notify_function) m_notify_function();
		}

		friend struct Accessor<T>;
	};


	template <class T>
	Handle<T>::Handle(const Owner<T>& owner)
		: Handle<T>(static_cast<const Handle<T>&>(owner))
	{}
	template <class T>
	Handle<T>::Handle(const Observer<T>& observer)
		: Handle<T>(static_cast<const Handle<T>&>(observer))
	{}

	template <class T>
	Handle<T>& Handle<T>::operator=(const Owner<T>& owner)
	{
		return (*this = static_cast<const Handle<T>&>(owner));
	}
	template <class T>
	Handle<T>& Handle<T>::operator=(const Observer<T>& observer)
	{
		return (*this = static_cast<const Handle<T>&>(observer));
	}
	template <class T>
	bool Handle<T>::operator==(const Owner<T>& owner) const
	{
		return (*this == static_cast<const Handle<T>&>(owner));
	}
	template <class T>
	bool Handle<T>::operator==(const Observer<T>& observer) const
	{
		return (*this == static_cast<const Handle<T>&>(observer));
	}


	template <class T>
	Observer<T>::Observer(const Owner<T>& owner)
		: Handle<T>(static_cast<const Handle<T>&>(owner))
		, m_notify_function(nullptr)
	{
		if (mp_accessor) mp_accessor->subscribeObserver(this);
	}
	template <class T>
	Observer<T>::Observer(
		const Handle<T>& handle,
		const std::function<void()>& f)
		: Handle<T>(handle)
		, m_notify_function(f)
	{
		if (mp_accessor) mp_accessor->subscribeObserver(this);
	}

	template <class T>
	Observer<T>& Observer<T>::operator=(const Owner<T>& owner)
	{
		return (*this = static_cast<const Handle<T>&>(owner));
	}
	template <class T>
	Observer<T>& Observer<T>::operator=(const Handle<T>& handle)
	{
		if (mp_accessor == handle.mp_accessor) return *this;

		if (mp_accessor)
		{
			mp_accessor->unsubscribeObserver(this);
			if (mp_accessor->decRefCount() == 0u)
				delete mp_accessor;
		}
		mp_accessor = handle.mp_accessor;
		if (mp_accessor)
		{
			mp_accessor->incRefCount();
			mp_accessor->subscribeObserver(this);
		}

		return *this;
	}

	template <class T>
	void Accessor<T>::notifyObservers()
	{
		for (Observer<T>* o : m_observers) o->notify();
	}
}

namespace std
{
	template <typename T>
	struct hash<RayZath::Engine::Handle<T>>
	{
		std::size_t operator()(const RayZath::Engine::Handle<T>& handle) const
		{
			return std::size_t(handle.accessor());
		}
	};
	template <typename T>
	struct less<RayZath::Engine::Handle<T>>
	{
		bool operator()(const RayZath::Engine::Handle<T>& left, const RayZath::Engine::Handle<T>& right) const
		{
			return left.accessor() < right.accessor();
		}
	};
}

#endif
