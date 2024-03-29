#ifndef OBJECT_CONTAINER_H
#define OBJECT_CONTAINER_H

#include "world_object.hpp"
#include "roho.hpp"

#include <map>
#include <string>

namespace RayZath::Engine
{
	template <class T>
	class ObjectContainer
		: public Updatable
	{
	private:
		static constexpr uint32_t sm_min_capacity = 4u;
		uint32_t m_count, m_capacity;
		Owner<T>* mp_owners;


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
		ObjectContainer(Updatable* updatable)
			: Updatable(updatable)
			, m_count(0u)
			, m_capacity(4u)
			, mp_owners(static_cast<Owner<T>*>(::operator new[](sizeof(Owner<T>)* m_capacity)))
		{}
		~ObjectContainer()
		{
			deallocate();
		}


	public:
		ObjectContainer& operator=(const ObjectContainer& other) = delete;
		ObjectContainer& operator=(ObjectContainer&& other)
		{
			if (this == &other)
				return *this;

			deallocate();

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
			for (uint32_t i = 0; i < this->count(); i++)
				if (auto& object = operator[](i); object && object->name() == object_name)
					return object;
			return Handle<T>{};
		}
		const Handle<T> operator[](const std::string& object_name) const
		{
			for (uint32_t i = 0; i < this->count(); i++)
				if (const auto& object = operator[](i); object && object->name() == object_name)
					return object;
			return Handle<T>{};
		}


	public:
		Handle<T> create(ConStruct<T> conStruct)
		{
			// check if has free space for new object, allocate
			// more memory otherwise
			growIfNecessary();

			// inplace construct new object via Owner
			new (&mp_owners[m_count]) Owner<T>(new T(this, std::move(conStruct)), m_count);

			stateRegister().MakeModified();
			return Handle<T>(mp_owners[m_count++]);
		}
		bool destroy(const Handle<T>& object)
		{
			return bool(object) ? destroy(object.accessor()->idx()) : false;
		}
		bool destroy(const uint32_t& idx)
		{
			if (idx >= count())
				return false;

			if (mp_owners[idx])
			{
				--m_count;
				if (idx < m_count)
				{
					// move the last owner to the selected to delete one
					mp_owners[idx] = std::move(mp_owners[m_count]);
					// set new idx for the moved object
					mp_owners[idx].accessor()->idx(idx);
				}

				// call destructor on last owner
				mp_owners[m_count].~Owner();

				shrinkIfNecessary();
				stateRegister().RequestUpdate();

				return true;
			}

			return false;
		}
		void destroyAll()
		{
			deallocate();

			m_capacity = 0u;
			m_count = 0u;

			stateRegister().RequestUpdate();
		}

		uint32_t count() const
		{
			return m_count;
		}
		uint32_t capacity() const
		{
			return	m_capacity;
		}
		bool empty() const
		{
			return count() == 0;
		}

		virtual void update() override
		{
			if (!stateRegister().RequiresUpdate()) return;

			for (uint32_t i = 0; i < m_count; ++i)
			{
				mp_owners[i]->update();
			}

			stateRegister().update();
		}

	private:
		void deallocate()
		{
			if (mp_owners)
			{
				for (uint32_t i = 0u; i < m_count; i++)
					mp_owners[i].~Owner();

				::operator delete[](mp_owners);
				mp_owners = nullptr;
			}
		}

		void growIfNecessary()
		{
			if (m_count >= m_capacity)
				resize(std::max(uint32_t(m_capacity * 2u), sm_min_capacity));
		}
		void shrinkIfNecessary()
		{
			if (m_count < m_capacity / 2u)
				resize(std::max(uint32_t(m_capacity / 2u), sm_min_capacity));
		}
		void resize(const uint32_t capacity)
		{
			if (m_capacity == capacity) return;

			// allocate new memory with new capacity
			Owner<T>* p_new_owners = static_cast<Owner<T>*>(::operator new[](sizeof(Owner<T>)* capacity));

			// move construct all components from the beginningt ocurrent count or capacity if
			// it happesn to be less than current count
			for (uint32_t i = 0u; i < std::min(m_count, capacity); ++i)
				new (&p_new_owners[i]) Owner<T>(std::move(mp_owners[i]));

			// call destructor for every component loacated in old memory
			for (uint32_t i = 0u; i < m_count; ++i)
				mp_owners[i].~Owner();

			// free old memory and assign member pointer to the new one
			::operator delete[](mp_owners);
			mp_owners = p_new_owners;

			// update cpacity and count values
			m_capacity = capacity;
			m_count = std::min(m_count, m_capacity);

			stateRegister().RequestUpdate();
		}
	};
}

#endif // !OBJECT_CONTAINER_H
