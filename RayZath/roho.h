#ifndef ROHO_H
#define ROHO_H

#include <assert.h>
#include <vector>
#include <functional>
#include <algorithm>
#include <memory>

namespace RayZath
{
	template <class T>
	struct Handle;
	template <class T>
	struct Owner;
	template <class T>
	struct Observer;


	template <class T>
	struct Resource
	{
	private:
		uint32_t m_id;
		T* m_data;


	public:
		Resource(const uint32_t& id, T* data)
			: m_id(id)
			, m_data(data)
		{}
		Resource(const Resource& right) = delete;
		Resource(Resource& right) = delete;
		~Resource()
		{
			delete m_data;
		}


	public:
		uint32_t GetId() const
		{
			return m_id;
		}
		void SetId(const uint32_t& id)
		{
			m_id = id;
		}
		T* GetData() const
		{
			return m_data;
		}
	};

	template <class T>
	struct Accessor
	{
	private:
		Resource<T>* mp_resource;
		size_t m_ref_count;
		std::vector<Observer<T>*> m_observers;


	public:
		Accessor(Resource<T>* resource)
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
		size_t IncRefCount()
		{
			return ++m_ref_count;
		}
		size_t DecRefCount()
		{
			assert(m_ref_count > 0u);
			return --m_ref_count;
		}
		void SubscribeObserver(Observer<T>* observer)
		{
			m_observers.push_back(observer);
		}
		void UnsubscribeObserver(const Observer<T>* observer)
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

		Resource<T>* GetResource()
		{
			return mp_resource;
		}
		const Resource<T>* GetResource() const
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
	private:
		void NotifyObservers();
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
			if (mp_accessor) mp_accessor->IncRefCount();
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
				if (mp_accessor->DecRefCount() == 0u)
					delete mp_accessor;
				mp_accessor = nullptr;
			}
		}

	public:
		Handle& operator=(const Handle& other)
		{
			if (this == &other) return *this;
			if (mp_accessor == other.mp_accessor) return *this;

			if (mp_accessor)
			{
				if (mp_accessor->DecRefCount() == 0u)
					delete mp_accessor;
			}
			mp_accessor = other.mp_accessor;
			if (mp_accessor) mp_accessor->IncRefCount();

			return *this;
		}
		Handle& operator=(Handle&& other) noexcept
		{
			if (this == &other) return *this;
			if (mp_accessor == other.mp_accessor) return *this;

			if (mp_accessor)
			{
				if (mp_accessor->DecRefCount() == 0u)
					delete mp_accessor;
			}
			mp_accessor = other.mp_accessor;
			other.mp_accessor = nullptr;

			return *this;
		}
		Handle& operator=(const Owner<T>& owner);
		Handle& operator=(const Observer<T>& observer);
		bool operator==(const Handle<T>& other) const
		{
			return (mp_accessor == other.mp_accessor);
		}
		bool operator==(const Owner<T>& owner) const;
		bool operator==(const Observer<T>& observer) const;
		T* operator->() const noexcept
		{
			return mp_accessor ? mp_accessor->GetResource()->GetData() : nullptr;
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
				if (mp_accessor->DecRefCount() == 0u)
					delete mp_accessor;
				mp_accessor = nullptr;
			}
		}
		const Resource<T>* GetResource() const
		{
			return mp_accessor ? mp_accessor->GetResource() : nullptr;
		}

		friend struct Owner<T>;
		friend struct Observer<T>;
	};

	template <class T>
	struct Owner : public Handle<T>
	{
	private:
		using Handle<T>::mp_accessor;
		using Handle<T>::Release;


	public:
		Owner()
		{}
		Owner(Resource<T>* resource)
		{
			mp_accessor = new Accessor<T>(resource);
		}
		Owner(const Owner& other) = delete;
		Owner(Owner&& other)
		{
			mp_accessor = other.mp_accessor;
			other.mp_accessor = nullptr;
		}
		~Owner()
		{
			if (mp_accessor) mp_accessor->DestroyObject();
		}


	public:
		void DestroyObject()
		{
			if (mp_accessor) mp_accessor->DestroyObject();
			Release();
		}
		void Reasign(Resource<T>* resource)
		{
			DestroyObject();
			mp_accessor = new Accessor<T>(resource);
		}
		Accessor<T>* GetAccessor()
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
		Observer()
		{}
		Observer(const std::function<void()>& f)
			: Handle<T>()
			, m_notify_function(f)
		{}
		Observer(const Observer& other)
			: Handle<T>(static_cast<const Handle<T>&>(other))
			, m_notify_function(other.m_notify_function)
		{
			if (mp_accessor) mp_accessor->SubscribeObserver(this);
		}
		Observer(Observer&& other)
		{
			// notify function
			m_notify_function = other.m_notify_function;
			other.m_notify_function = nullptr;

			// accessor
			if (other.mp_accessor)
			{
				other.mp_accessor->UnsubscribeObserver(&other);
				other.mp_accessor->SubscribeObserver(this);
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
			if (mp_accessor) mp_accessor->UnsubscribeObserver(this);
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
				mp_accessor->UnsubscribeObserver(this);
				if (mp_accessor->DecRefCount() == 0u) delete mp_accessor;
			}
			mp_accessor = other.mp_accessor;
			if (mp_accessor)
			{
				mp_accessor->IncRefCount();
				mp_accessor->SubscribeObserver(this);
			}

			return *this;
		}
		Observer& operator=(Observer&& other)
		{
			if (this == &other) return *this;
			if (mp_accessor == other.mp_accessor) return *this;

			// notify function
			m_notify_function = other.m_notify_function;
			other.m_notify_function = nullptr;

			// accessor
			if (mp_accessor)
			{
				mp_accessor->UnsubscribeObserver(this);
				if (mp_accessor->DecRefCount() == 0u) delete mp_accessor;
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
		Observer& operator=(const Handle<T>& handle);


	public:
		void SetNotifyFunction(const std::function<void()>& f)
		{
			m_notify_function = f;
		}
		const std::function<void()>& GetNotifyFunction()
		{
			return m_notify_function;
		}
		void Release()
		{
			if (mp_accessor) mp_accessor->UnsubscribeObserver(this);
			Handle<T>::Release();
		}
	private:
		void Notify()
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
		if (mp_accessor) mp_accessor->SubscribeObserver(this);
	}
	template <class T>
	Observer<T>::Observer(
		const Handle<T>& handle,
		const std::function<void()>& f)
		: Handle<T>(handle)
		, m_notify_function(f)
	{
		if (mp_accessor) mp_accessor->SubscribeObserver(this);
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
			mp_accessor->UnsubscribeObserver(this);
			if (mp_accessor->DecRefCount() == 0u)
				delete mp_accessor;
		}
		mp_accessor = handle.mp_accessor;
		if (mp_accessor)
		{
			mp_accessor->IncRefCount();
			mp_accessor->SubscribeObserver(this);
		}

		return *this;
	}

	template <class T>
	void Accessor<T>::NotifyObservers()
	{
		for (Observer<T>* o : m_observers) o->Notify();
	}
}

#endif