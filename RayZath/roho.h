#ifndef ROHO_H
#define ROHO_H

#include <assert.h>
#include <vector>
#include <functional>
#include <algorithm>

namespace RayZath
{
	template <class T>
	class Owner;
	template <class T>
	class Handle;
	template <class T>
	class Observer;


	template <class T>
	struct Accessor
	{
	private:
		T* mp_resource;
		size_t m_ref_count;
		std::vector<Observer<T>*> m_observers;

	public:
		Accessor(T* resource)
			: mp_resource(resource)
			, m_ref_count(1u)
		{}
		Accessor(const Accessor& other) = delete;
		Accessor(Accessor&& other) = delete;
		~Accessor()
		{
			assert(m_ref_count == 0u);
			assert(m_observers.size() == 0u);

			DestroyObject();
		}


	public:
		Accessor& operator=(const Accessor& other) = delete;
		Accessor& operator=(Accessor&& other) = delete;
		explicit operator bool() const noexcept
		{
			return mp_resource != nullptr;
		}


	public:
		void SubscribeHolder()
		{
			++m_ref_count;
		}
		void UnsubscribeHolder()
		{
			assert(m_ref_count > 0u);
			--m_ref_count;
		}
		void SubscribeObserver(Observer<T>* observer)
		{
			m_observers.push_back(observer);
		}
		void UnsubscribeObserver(const Observer<T>* observer)
		{
			auto iter = std::find(m_observers.begin(), m_observers.end(), observer);
			if (iter != m_observers.end())
				m_observers.erase(iter);
			else
				assert(false && "Unrecognized observer tried to unsubscribe!");
		}
		size_t HolderCount()
		{
			return m_ref_count;
		}
		size_t ObserverCount()
		{
			return m_observers.size();
		}
		size_t AllRefCount()
		{
			return HolderCount() + ObserverCount();
		}
		T* Get()
		{
			return mp_resource;
		}
		const T* Get() const
		{
			return mp_resource;
		}
		void DestroyObject()
		{
			if (mp_resource)
			{
				delete mp_resource;
				mp_resource = nullptr;
				NotifyObservers();
			}
		}
		void SetResource(T* resource)
		{
			mp_resource = resource;
		}
	private:
		void NotifyObservers();
	};


	template <class T>
	struct Owner
	{
	private:
		Accessor<T>* mp_accessor;


	public:
		Owner()
			: mp_accessor(nullptr)
		{}
		Owner(T* resource)
		{
			mp_accessor = new Accessor(resource);
		}
		Owner(const Owner& other) = delete;
		Owner(Owner&& other)
		{
			mp_accessor = other.mp_accessor;
			other.mp_accessor = nullptr;
		}
		~Owner()
		{
			if (mp_accessor)
			{
				mp_accessor->DestroyObject();
				mp_accessor->UnsubscribeHolder();
				if (mp_accessor->AllRefCount() == 0u) delete mp_accessor;
			}
			mp_accessor = nullptr;
		}


	public:
		T* operator->()
		{
			return mp_accessor ? mp_accessor->Get() : nullptr;
		}
		const T* operator->() const
		{
			return mp_accessor ? mp_accessor->Get() : nullptr;
		}

		explicit operator bool() const
		{
			return mp_accessor ? bool(*mp_accessor) : false;
		}

	public:
		void DestroyObject()
		{
			if (mp_accessor) mp_accessor->DestroyObject();
		}
		Accessor<T>* GetAccessor()
		{
			return mp_accessor;
		}


		friend class Handle<T>;
		friend class Observer<T>;
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
			if (mp_accessor) mp_accessor->SubscribeHolder();
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
				mp_accessor->UnsubscribeHolder();
				if (mp_accessor->AllRefCount() == 0u) delete mp_accessor;
			}
		}

	public:
		Handle& operator=(const Handle& other)
		{
			if (this == &other) return *this;
			if (mp_accessor == other.mp_accessor) return *this;

			if (mp_accessor)
			{
				mp_accessor->UnsubscribeHolder();
				if (mp_accessor->AllRefCount() == 0u) delete mp_accessor;
			}

			mp_accessor = other.mp_accessor;
			if (mp_accessor) mp_accessor->SubscribeHolder();

			return *this;
		}
		Handle& operator=(Handle&& other) noexcept
		{
			if (this == &other) return *this;
			if (mp_accessor == other.mp_accessor) return *this;

			if (mp_accessor)
			{
				mp_accessor->UnsubscribeHolder();
				if (mp_accessor->AllRefCount() == 0u) delete mp_accessor;
			}

			mp_accessor = other.mp_accessor;
			other.mp_accessor = nullptr;

			return *this;
		}
		Handle& operator=(const Owner<T>& owner);
		Handle& operator=(const Observer<T>& observer);

		T* operator->()
		{
			return mp_accessor ? mp_accessor->Get() : nullptr;
		}
		const T* operator->() const noexcept
		{
			return mp_accessor ? mp_accessor->Get() : nullptr;
		}
		explicit operator bool() const noexcept
		{
			return mp_accessor ? bool(*mp_accessor) : false;
		}


	public:
		void Release()
		{
			if (mp_accessor)
			{
				mp_accessor->UnsubscribeHolder();
				if (mp_accessor->AllRefCount() == 0u) delete mp_accessor;
			}
			mp_accessor = nullptr;
		}

		friend class Observer<T>;
	};


	template <class T>
	struct Observer
	{
	private:
		std::function<void()> m_notify_function;
		Accessor<T>* mp_accessor;


	public:
		Observer()
			: mp_accessor(nullptr)
		{}
		Observer(const std::function<void()>& f)
			: mp_accessor(nullptr)
			, m_notify_function(f)
		{}
		Observer(const Observer& other)
			: m_notify_function(other.m_notify_function)
			, mp_accessor(other.mp_accessor)
		{
			if (mp_accessor) mp_accessor->SubscribeObserver(this);
		}
		Observer(Observer&& other)
		{
			m_notify_function = other.m_notify_function;
			other.m_notify_function = nullptr;

			if (other.mp_accessor)
			{
				other.mp_accessor->UnsubscribeObserver(&other);
				other.mp_accessor->SubscribeObserver(this);
			}
			mp_accessor = other.mp_accessor;
			other.mp_accessor = nullptr;
		}
		Observer(const Owner<T>& owner);
		Observer(const Handle<T>& holder);
		Observer(const Handle<T>& holder, const std::function<void()>& f);
		~Observer()
		{
			if (mp_accessor)
			{
				mp_accessor->UnsubscribeObserver(this);
				if (mp_accessor->AllRefCount() == 0u) delete mp_accessor;
			}
		}


	public:
		Observer& operator=(const Observer& other)
		{
			if (this == &other) return *this;
			if (mp_accessor == other.mp_accessor) return *this;

			m_notify_function = other.m_notify_function;

			if (mp_accessor)
			{
				mp_accessor->UnsubscribeObserver(this);
				if (mp_accessor->AllRefCount() == 0u) delete mp_accessor;
			}
			mp_accessor = other.mp_accessor;
			if (mp_accessor) mp_accessor->SubscribeObserver(this);

			return *this;
		}
		Observer& operator=(Observer&& other)
		{
			if (this == &other) return *this;
			if (mp_accessor == other.mp_accessor) return *this;

			m_notify_function = other.m_notify_function;
			other.m_notify_function = nullptr;

			if (mp_accessor)
			{
				mp_accessor->UnsubscribeObserver(this);
				if (mp_accessor->AllRefCount() == 0u) delete mp_accessor;
			}
			if (other.mp_accessor)
			{
				other.mp_accessor->UnsubscribeObserver(&other);
				other.mp_accessor->SubscribeObserver(this);
			}
			mp_accessor = other.mp_accessor;
			other.mp_accessor = nullptr;

			return *this;
		}
		Observer& operator=(const Owner<T>& owner);
		Observer& operator=(const Handle<T>& holder);

		T* operator->()
		{
			return mp_accessor ? mp_accessor->Get() : nullptr;
		}
		const T* operator->() const
		{
			return mp_accessor ? mp_accessor->Get() : nullptr;
		}
		explicit operator bool() const
		{
			return mp_accessor ? bool(*mp_accessor) : false;
		}


	public:
		void SetNotifyFunction(const std::function<void()>& f)
		{
			m_notify_function = f;
		}
	private:
		void Notify()
		{
			if (m_notify_function) m_notify_function();
		}

		friend class Accessor<T>;
	};



	template <class T>
	Handle<T>::Handle(const Owner<T>& owner)
	{
		mp_accessor = owner.mp_accessor;
		if (mp_accessor) mp_accessor->SubscribeHolder();
	}
	template <class T>
	Handle<T>::Handle(const Observer<T>& observer)
	{
		mp_accessor = observer.mp_accessor;
		if (mp_accessor) mp_accessor->SubscribeHolder();
	}

	template <class T>
	Handle<T>& Handle<T>::operator=(const Owner<T>& owner)
	{
		if (mp_accessor)
		{
			mp_accessor->UnsubscribeHolder();
			if (mp_accessor->AllRefCount() == 0u) delete mp_accessor;
		}

		mp_accessor = owner.mp_accessor;
		if (mp_accessor) mp_accessor->SubscribeHolder();

		return *this;
	}
	template <class T>
	Handle<T>& Handle<T>::operator=(const Observer<T>& observer)
	{
		if (mp_accessor)
		{
			mp_accessor->UnsubscribeHolder();
			if (mp_accessor->AllRefCount() == 0u) delete mp_accessor;
		}

		mp_accessor = observer.mp_accessor;
		if (mp_accessor) mp_accessor->SubscribeHolder();

		return *this;
	}


	template <class T>
	Observer<T>::Observer(const Owner<T>& owner)
		: m_notify_function(nullptr)
	{
		mp_accessor = owner.mp_accessor;
		if (mp_accessor) mp_accessor->SubscribeObserver(this);
	}
	template <class T>
	Observer<T>::Observer(const Handle<T>& holder)
		: m_notify_function(nullptr)
	{
		mp_accessor = holder.mp_accessor;
		if (mp_accessor) mp_accessor->SubscribeObserver(this);
	}
	template <class T>
	Observer<T>::Observer(
		const Handle<T>& holder,
		const std::function<void()>& f)
		: m_notify_function(f)
	{
		mp_accessor = holder.mp_accessor;
		if (mp_accessor) mp_accessor->SubscribeObserver(this);
	}

	template <class T>
	Observer<T>& Observer<T>::operator=(const Owner<T>& owner)
	{
		if (mp_accessor == owner.mp_accessor) return *this;

		if (mp_accessor)
		{
			mp_accessor->UnsubscribeObserver(this);
			if (mp_accessor->AllRefCount() == 0u) delete mp_accessor;
		}
		mp_accessor = owner.mp_accessor;
		if (mp_accessor) mp_accessor->SubscribeObserver(this);

		return *this;
	}
	template <class T>
	Observer<T>& Observer<T>::operator=(const Handle<T>& holder)
	{
		if (mp_accessor == holder.mp_accessor) return *this;

		if (mp_accessor)
		{
			mp_accessor->UnsubscribeObserver(this);
			if (mp_accessor->AllRefCount() == 0u) delete mp_accessor;
		}
		mp_accessor = holder.mp_accessor;
		if (mp_accessor) mp_accessor->SubscribeObserver(this);

		return *this;
	}

	template <class T>
	void Accessor<T>::NotifyObservers()
	{
		for (Observer<T>* o : m_observers) o->Notify();
	}
}

#endif