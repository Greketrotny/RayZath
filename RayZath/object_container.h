#ifndef OBJECT_CONTAINER_H
#define OBJECT_CONTAINER_H

#include "world_object.h"

namespace RayZath
{
	class ObjectCreator
	{
	protected:
		template <class T, typename... Params> T* Create(Params... params)
		{
			return new T(params...);
		}
		template <class T, typename... Params> void CreateInPlace(T* ptr, Params... params)
		{
			new (ptr) T(params...);
		}
		template <class T> static void Destroy(T*& object)
		{
			if (object) delete object;
			object = nullptr;
		}
		template <class T> static void Destruct(T& object)
		{
			object.~T();
		}
	};

	template <class T> struct ObjectContainer 
		: public Updatable
		, public ObjectCreator
	{
	private:
		uint32_t m_count, m_capacity;
		T* mp_storage = nullptr;
		T** mpp_storage_ptr = nullptr;


	public:
		ObjectContainer(Updatable* updatable, uint32_t capacity = 16u)
			: Updatable(updatable)
		{
			this->m_count = 0u;
			this->m_capacity = std::max(capacity, uint32_t(2));

			mp_storage = (T*)malloc(this->m_capacity * sizeof(T));
			mpp_storage_ptr = (T**)malloc(this->m_capacity * sizeof(T*));
			for (uint32_t i = 0u; i < this->m_capacity; ++i)
				mpp_storage_ptr[i] = nullptr;
		}
		~ObjectContainer()
		{
			for (uint32_t i = 0u; i < m_capacity; ++i)
			{
				if (mpp_storage_ptr[i])
				{
					ObjectCreator::Destruct(mp_storage[i]);
					mpp_storage_ptr[i] = nullptr;
				}
			}

			if (mp_storage) free(mp_storage);
			mp_storage = nullptr;
			if (mpp_storage_ptr) free(mpp_storage_ptr);
			mpp_storage_ptr = nullptr;

			m_count = 0u;
			m_capacity = 0u;
		}


	public:
		T* operator[](uint32_t index)
		{
			return mpp_storage_ptr[index];
		}
		const T* operator[](uint32_t index) const
		{
			return mpp_storage_ptr[index];
		}
		

	public:
		T* CreateObject(const ConStruct<T>& conStruct)
		{
			if (m_count >= m_capacity)
				return nullptr;

			T* newObject = nullptr;
			for (uint32_t i = 0u; i < m_capacity; ++i)
			{
				if (!mpp_storage_ptr[i])
				{
					ObjectCreator::CreateInPlace<T>(&mp_storage[i], i, this, conStruct);
					mpp_storage_ptr[i] = &mp_storage[i];
					newObject = &mp_storage[i];
					++m_count;
					return newObject;
				}
			}
			return newObject;
		}
		bool DestroyObject(const T* object)
		{
			if (m_count == 0u || object == nullptr)
				return false;

			for (uint32_t i = 0u; i < m_capacity; ++i)
			{
				if (mpp_storage_ptr[i] && object == &mp_storage[i])
				{
					ObjectCreator::Destruct(mp_storage[i]);
					mpp_storage_ptr[i] = nullptr;
					--m_count;
					return true;
				}
			}
			return false;
		}
		void DestroyAllObjects()
		{
			for (uint32_t i = 0u; i < m_capacity; ++i)
			{
				if (mpp_storage_ptr[i])
				{
					ObjectCreator::Destruct(mp_storage[i]);
					mpp_storage_ptr[i] = nullptr;
				}
			}
			m_count = 0u;
		}

		uint32_t GetCount() const
		{
			return m_count;
		}
		uint32_t GetCapacity() const
		{
			return	m_capacity;
		}


		void Update() override
		{
			if (!GetStateRegister().RequiresUpdate()) return;

			for (uint32_t i = 0; i < m_capacity; ++i)
			{
				if (mpp_storage_ptr[i])
					mpp_storage_ptr[i]->Update();
			}

			GetStateRegister().Update();
		}
	};
}

#endif // !OBJECT_CONTAINER_H