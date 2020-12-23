#ifndef OBJECT_CONTAINER_H
#define OBJECT_CONTAINER_H

#include "world_object.h"
#include "roho.h"

namespace RayZath
{
	template <class T> 
	class ObjectContainer 
		: public Updatable
	{
	private:
		uint32_t m_count, m_capacity;
		Owner<T>* mp_owners;


	public:
		ObjectContainer(
			Updatable* updatable, 
			const uint32_t capacity = 16u)
			: Updatable(updatable)
			, m_count(0u)
			, m_capacity(std::max(capacity, 2u))
			, mp_owners(new Owner<T>[m_capacity]())
		{}
		~ObjectContainer()
		{
			delete[] mp_owners;
		}


	public:
		const Handle<T>& operator[](const uint32_t& index) const
		{
			return static_cast<const Handle<T>&>(mp_owners[index]);
		}
		Handle<T> operator[](const uint32_t& index)
		{
			return mp_owners[index];
		}
		

	public:
		Handle<T> CreateObject(const ConStruct<T>& conStruct)
		{
			if (m_count >= m_capacity) return Handle<T>();

			for (uint32_t i = 0u; i < m_capacity; ++i)
			{
				if (!mp_owners[i])
				{
					mp_owners[i].Reasign(new Resource<T>(i, new T(this, conStruct)));
					++m_count;
					return Handle<T>(mp_owners[i]);
				}
			}
			return Handle<T>();
		}
		bool DestroyObject(const Handle<T>& object)
		{
			if (m_count == 0u || !object)
				return false;

			for (uint32_t i = 0u; i < m_capacity; ++i)
			{
				if (mp_owners[i])
				{
					if (mp_owners[i] == object)
					{
						mp_owners[i].DestroyObject();
						--m_count;
						return true;
					}
				}
			}

			return false;
		}
		void DestroyAllObjects()
		{
			for (uint32_t i = 0u; i < m_capacity; ++i)
			{
				mp_owners[i].DestroyObject();
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
				if (mp_owners[i]) mp_owners[i]->Update();
			}

			GetStateRegister().Update();
		}
	};
}

#endif // !OBJECT_CONTAINER_H