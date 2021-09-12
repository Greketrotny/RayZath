#ifndef OBJECT_CONTAINER_H
#define OBJECT_CONTAINER_H

#include "world_object.h"
#include "roho.h"

#include <map>
#include <string>

namespace RayZath
{
	template <class T>
	class ObjectContainer
		: public Updatable
	{
	private:
		uint32_t m_count, m_capacity;
		Owner<T>* mp_owners;

		std::map<std::string, Handle<T>> m_name_map;
		uint64_t m_name_counter;


	public:
		ObjectContainer(const ObjectContainer& other) = delete;
		ObjectContainer(ObjectContainer&& other)
			: Updatable(std::move(other))
			, m_count(other.m_count)
			, m_capacity(other.m_capacity)
			, mp_owners(other.mp_owners)
		{
			other.m_count = 0u;
			other.m_capacity = 0u;
			other.mp_owners = nullptr;
		}
		ObjectContainer(
			Updatable* updatable,
			const uint32_t capacity = 16u)
			: Updatable(updatable)
			, m_count(0u)
			, m_capacity(std::max(capacity, 2u))
			, mp_owners(new Owner<T>[m_capacity]())
			, m_name_counter(0u)
		{}
		~ObjectContainer()
		{
			if (mp_owners)
				delete[] mp_owners;
			mp_owners = nullptr;
		}


	public:
		ObjectContainer& operator=(const ObjectContainer& other) = delete;
		ObjectContainer& operator=(ObjectContainer&& other)
		{
			if (this == &other) return *this;

			m_name_map = std::move(other.m_name_map);

			if (mp_owners)
				delete[] mp_owners;

			m_count = other.m_count;
			m_capacity = other.m_capacity;
			mp_owners = other.mp_owners;

			other.m_count = 0u;
			other.m_capacity = 0u;
			other.mp_owners = nullptr;

			return *this;
		}
		const Handle<T>& operator[](const uint32_t& index) const
		{
			return static_cast<const Handle<T>&>(mp_owners[index]);
		}
		const Handle<T>& operator[](const uint32_t& index)
		{
			return static_cast<const Handle<T>&>(mp_owners[index]);
		}
		const Handle<T> operator[](const std::string& object_name)
		{
			auto result = m_name_map.find(object_name);
			return (result == m_name_map.end()) ? Handle<T>() : result->second;
		}
		const Handle<T> operator[](const std::string& object_name) const
		{
			auto result = m_name_map.find(object_name);
			return (result == m_name_map.end()) ? Handle<T>() : result->second;
		}


	public:
		Handle<T> Create(const ConStruct<T>& conStruct)
		{
			if (m_count >= m_capacity) return Handle<T>();

			auto result = m_name_map.find(conStruct.name);
			if (result != m_name_map.end())
				return Handle<T>();

			for (uint32_t i = 0u; i < m_capacity; ++i)
			{
				if (!mp_owners[i])
				{
					mp_owners[i].Reasign(new Resource<T>(i, new T(this, conStruct)));
					++m_count;

					m_name_map[mp_owners[i]->GetName()] = mp_owners[i];
					GetStateRegister().MakeModified();
					return Handle<T>(mp_owners[i]);
				}
			}
			return Handle<T>();
		}
		bool Destroy(const Handle<T>& object)
		{
			if (m_count == 0u || !object)
				return false;

			if (object == mp_owners[object.GetResource()->GetId()])
			{
				m_name_map.erase(object->GetName());

				mp_owners[object.GetResource()->GetId()].Destroy();
				--m_count;
				GetStateRegister().MakeModified();
				return true;
			}

			return false;
		}
		bool Destroy(const uint32_t& index)
		{
			if (index >= GetCapacity()) return false;

			if (mp_owners[index])
			{
				m_name_map.erase(mp_owners[index]->GetName())
					;
				mp_owners[index].Destroy();
				--m_count;
				GetStateRegister().MakeModified();
				return true;
			}

			return false;
		}
		void DestroyAll()
		{
			for (uint32_t i = 0u; i < m_capacity; ++i)
			{
				mp_owners[i].Destroy();
			}
			m_count = 0u;
			m_name_map.clear();
			GetStateRegister().MakeModified();
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