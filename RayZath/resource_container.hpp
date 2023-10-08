#ifndef RESOURCE_CONTAINER_HPP
#define RESOURCE_CONTAINER_HPP

#include "world_object.hpp"
#include "shared_resource.hpp"

#include <shared_mutex>
#include <memory>

namespace RayZath::Engine
{
	template <class T>
	class ResourceContainer
		: public Updatable
	{
	public:
		//static_assert(std::is_nothrow_move_constructible_v<T>&& std::is_nothrow_move_assignable_v<T>);
		using size_t = uint32_t;
		using idx_t = uint32_t;

		template <typename IT>
		class Iterator
		{
		public:
			using value_type = IT;
			using difference_type = std::ptrdiff_t;
			using pointer = IT*;
			using reference = IT&;
			using iterator_category = std::input_iterator_tag;

		private:
			IT* mp_object;


		public:
			explicit Iterator(IT* p_object) noexcept
				: mp_object(p_object)
			{}

			IT& operator*() noexcept { return *mp_object; }
			const IT& operator*() const noexcept { return *mp_object; }
			bool operator==(const Iterator& other) const noexcept { return this->mp_object == other.mp_object; }
			bool operator!=(const Iterator& other) const noexcept { return !(*this == other); }
			Iterator& operator++() noexcept
			{
				++mp_object;
				return *this;
			}
		};

		static constexpr size_t m_resize_factor = 2;

	private:
		size_t m_count = 0, m_capacity = 0;
		T* mp_objects = nullptr;
		Owner<T>* mp_owners = nullptr;
		mutable std::shared_mutex m_mtx;


	public:
		ResourceContainer(const ResourceContainer& other) = delete;
		ResourceContainer(ResourceContainer&& other) noexcept
			: Updatable(std::move(other))
			, m_count(std::exchange(other.m_count, 0))
			, m_capacity(std::exchange(other.m_capacity, 0))
			, mp_owners(std::exchange(other.mp_owners, nullptr))
		{}
		ResourceContainer(Updatable* updatable)
			: Updatable(updatable)
		{}
		~ResourceContainer()
		{
			deallocate();
		}

		ResourceContainer& operator=(const ResourceContainer& other) = delete;
		ResourceContainer& operator=(ResourceContainer&& other) noexcept
		{
			if (this == &other)
				return *this;

			deallocate();

			m_count = std::exchange(other.m_count, 0);
			m_capacity = std::exchange(other.m_capacity, 0);
			mp_owners = std::exchange(other.mp_owners, nullptr);

			return *this;
		}

		const T& operator[](idx_t idx) const noexcept
		{
			return mp_objects[idx];
		}
		T& operator[](idx_t idx) noexcept
		{
			return mp_objects[idx];
		}

		Iterator<T> begin() noexcept { return Iterator(mp_objects); }
		Iterator<T> end() noexcept { return Iterator(mp_objects + m_count); }
		Iterator<const T> begin() const noexcept { return Iterator<const T>(mp_objects); }
		Iterator<const T> end() const noexcept { return Iterator<const T>(mp_objects + m_count); }

		size_t count() const
		{
			return m_count;
		}
		size_t capacity() const
		{
			return	m_capacity;
		}
		bool empty() const
		{
			return count() == 0;
		}
		auto& mtx() const noexcept
		{
			return m_mtx;
		}

		Handle<T> add(T&& object)
		{
			grow();
			const idx_t new_idx = m_count;
			new (&mp_owners[new_idx]) Owner<T>{};
			new (&mp_objects[new_idx]) T(std::forward<T>(object));
			m_count++;

			stateRegister().MakeModified();
			return handle(new_idx);
		}
		Handle<T> add(ConStruct<T>&& construct)
		{
			return add(T(this, std::move(construct)));
		}
		Handle<T> handle(const idx_t idx)
		{
			RZAssertCore(idx < m_count, "Out of bound access.");
			auto& owner = mp_owners[idx];
			if (auto handle = owner.handle(); handle)
			{
				return handle;
			}

			auto accessor = std::make_shared<Owner<T>::accessor_t>(mp_objects + idx, idx, m_mtx);
			owner = Owner<T>(accessor);
			return Handle<T>(std::move(accessor));
		}
		Handle<T> handle(const T& object)
		{
			RZAssertCore(&object >= mp_objects && &object < mp_objects + m_count, "Object out of range.");
			return handle(idx_t((&object) - mp_objects));
		}
		bool destroy(const Handle<T>& object)
		{
			return bool(object) ? destroy(object.accessor()->idx()) : false;
		}
		bool destroy(idx_t idx)
		{
			if (idx >= m_count)
				return false;

			std::destroy_at(mp_objects + idx);
			std::destroy_at(mp_owners + idx);

			if (const bool was_last = idx + 1 == m_count; !was_last)
			{
				const auto last_idx = m_count - 1;
				new (&mp_objects[idx]) T(std::move(mp_objects[last_idx]));
				new (&mp_owners[idx]) Owner<T>(std::move(mp_owners[last_idx]), mp_objects + idx, idx);

				std::destroy_at(mp_owners + last_idx);
				std::destroy_at(mp_objects + last_idx);
			}

			--m_count;
			shrink();
			stateRegister().RequestUpdate();

			return true;
		}
		void destroyAll()
		{
			deallocate();

			m_capacity = 0u;
			m_count = 0u;

			stateRegister().RequestUpdate();
		}


		virtual void update() override
		{
			if (!stateRegister().RequiresUpdate()) return;

			for (auto& object : *this)
				object.update();

			stateRegister().update();
		}

	private:
		void deallocate()
		{
			deallocateOwners();
			deallocateObjects();
		}
		void deallocateOwners() noexcept
		{
			if (mp_owners)
			{
				std::destroy_n(mp_owners, m_count);
				::operator delete[](mp_owners);
				mp_owners = nullptr;
			}
		}
		void deallocateObjects() noexcept
		{
			if (mp_objects)
			{
				std::destroy_n(mp_objects, m_count);
				::operator delete[](mp_objects);
				mp_objects = nullptr;
			}
		}

		void grow()
		{
			if (m_count >= m_capacity)
			{
				auto new_capacity = std::numeric_limits<size_t>::max() / m_resize_factor < m_capacity ?
					std::numeric_limits<size_t>::max() :
					m_capacity * m_resize_factor;
				new_capacity = std::max(size_t(1), new_capacity);
				resize(new_capacity);
			}
		}
		void shrink()
		{
			if (m_count < m_capacity / (m_resize_factor * m_resize_factor))
			{
				resize(m_capacity / m_resize_factor);
			}
		}
		void resize(const size_t capacity)
		{
			if (m_capacity == capacity) return;
			if (capacity == 0)
			{
				deallocate();
				m_capacity = capacity;
				return;
			}

			std::unique_ptr<Owner<T>[]> new_owners((Owner<T>*)::operator new[](sizeof(Owner<T>)* capacity));
			std::unique_ptr<T[]> new_objects((T*)::operator new[](sizeof(T)* capacity));

			stateRegister().RequestUpdate();

			// move construct all components from the beginning of current count or capacity if
			// it happes to be less than current count
			for (idx_t i = 0; i < std::min(m_count, capacity); ++i)
			{
				new (&new_objects[i]) T(std::move(mp_objects[i]));
				new (&new_owners[i]) Owner<T>(std::move(mp_owners[i]), &new_objects[i], i);
			}

			deallocate();

			mp_objects = new_objects.release();
			mp_owners = new_owners.release();

			m_capacity = capacity;
			m_count = std::min(m_count, m_capacity);
		}
	};
}

#endif // !OBJECT_CONTAINER_H
