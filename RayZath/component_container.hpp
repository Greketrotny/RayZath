#ifndef COMPONENT_CONTAINER_H
#define COMPONENT_CONTAINER_H

#include "render_parts.hpp"
#include "rzexception.hpp"
#include "mesh_component.hpp"

#include <algorithm>

namespace RayZath::Engine
{
	struct MeshStructure;

	template <typename T, bool WithBVH = std::is_same_v<T, Triangle>>
	struct ComponentContainer;

	template <typename T>
	struct ComponentContainer<T, false>
		: public Updatable
	{
	private:
		static constexpr uint32_t sm_min_capacity = 4u;
		uint32_t m_capacity, m_count;
		T* mp_memory;
	protected:
		const MeshStructure& mr_mesh_structure;
	public:
		static constexpr uint32_t sm_npos = std::numeric_limits<uint32_t>::max();


	public:
		ComponentContainer(
			Updatable* parent,
			const MeshStructure& structure,
			const uint32_t& capacity = 64u)
			: Updatable(parent)
			, mr_mesh_structure(structure)
			, m_capacity(capacity)
			, m_count(0u)
		{
			mp_memory = static_cast<T*>(::operator new[](sizeof(T)* m_capacity));
		}
		~ComponentContainer()
		{
			// destroy each component stored in container
			for (uint32_t i = 0u; i < m_count; i++)
				mp_memory[i].~T();

			::operator delete[](mp_memory);
			mp_memory = nullptr;

			m_capacity = 0u;
			m_count = 0u;
		}


	public:
		T& operator[](const uint32_t& index)
		{
			return mp_memory[index];
		}
		const T& operator[](const uint32_t& index) const
		{
			return mp_memory[index];
		}


	public:
		const uint32_t& count() const
		{
			return m_count;
		}
		const uint32_t& capacity() const
		{
			return m_capacity;
		}
		static uint32_t endPos()
		{
			return sm_npos;
		}

		uint32_t add(const T& new_component)
		{
			growIfNecessary();
			new (&mp_memory[m_count]) T(new_component);

			stateRegister().RequestUpdate();
			return m_count++;
		}
		uint32_t add(T&& new_component)
		{
			growIfNecessary();
			new (&mp_memory[m_count]) T(std::move(new_component));

			stateRegister().RequestUpdate();
			return m_count++;
		}
		void removeAll()
		{
			resize(0u);
			stateRegister().RequestUpdate();
		}
	private:

		// resize capacity if is equal to the current count (memory is full) or 
		// count is below a half of capacity
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
		void resize(uint32_t capacity)
		{
			if (m_capacity == capacity) return;

			// allocate new memory buffer with new capacity
			T* new_memory = static_cast<T*>(::operator new[](sizeof(T)* capacity));

			// move construct all components from the beginning to current count or capacity if 
			// it happens to be less than current count
			for (uint32_t i = 0u; i < std::min(m_count, capacity); ++i)
				new (&new_memory[i]) T(std::move(mp_memory[i]));

			// call destructor for every component located in old memory
			for (uint32_t i = 0u; i < m_count; ++i)
				mp_memory[i].~T();

			// free old buffer and assing member pointer to the new one
			::operator delete[](mp_memory);
			mp_memory = new_memory;

			// update capacity and count values
			m_capacity = capacity;
			m_count = std::min(m_count, m_capacity);

			stateRegister().RequestUpdate();
		}
	};

	template <class T, uint32_t leaf_size = 8u, uint32_t max_depth = 31u>
	struct ComponentTreeNode
	{
	public:
		static_assert(leaf_size != 0);
		struct Children
		{
			enum class PartitionType : uint8_t
			{
				X = 2,
				Y = 1,
				Z = 0,
				Size = 3
			};

			ComponentTreeNode first, second;
			PartitionType type;

			Children(ComponentTreeNode&& f, ComponentTreeNode&& s, PartitionType t)
				: first(std::move(f))
				, second(std::move(s))
				, type(t) {}
		};

		using objects_t = std::vector<T*>;
		using objects_iterator = typename objects_t::iterator;
	private:
		std::unique_ptr<Children> m_children;
		objects_t m_objects;
		BoundingBox m_bb;

	public:
		ComponentTreeNode() = default;
		ComponentTreeNode(const MeshStructure& structure, objects_t&& components, const uint32_t depth = 0)
			: ComponentTreeNode(
				structure,
				[&components, &structure]()
				{
					BoundingBox bb;
					if (!components.empty()) bb = components[0]->boundingBox(structure);
					for (const auto& component : components)
						bb.extendBy(component->boundingBox(structure));
					return bb;
				}(),
					components.begin(), components.end(),
					depth)
		{}
	private:
		ComponentTreeNode(
			const MeshStructure& structure,
			const BoundingBox& bb,
			const objects_iterator begin, const objects_iterator end,
			const uint32_t depth)
			: m_bb(bb)
		{
			construct(structure, begin, end, depth);
			fitBoundingBox(structure);
		}


	public:
		const std::unique_ptr<Children>& children() const
		{
			return m_children;
		}
		const objects_t& objects() const
		{
			return m_objects;
		}
		const BoundingBox& boundingBox() const
		{
			return m_bb;
		}

		void clear()
		{
			m_children.release();
			m_objects.clear();
		}
		uint32_t treeSize() const
		{
			return children() ? children()->first.treeSize() + children()->second.treeSize() + 1 : 1;
		}
		bool isLeaf() const
		{
			return !m_children;
		}
	private:
		BoundingBox fitBoundingBox(const MeshStructure& structure)
		{
			m_bb = BoundingBox();

			if (isLeaf())
			{
				if (!objects().empty())
				{
					m_bb = objects()[0]->boundingBox(structure);
					for (size_t i = 1; i < objects().size(); i++)
					{
						m_bb.extendBy(objects()[i]->boundingBox(structure));
					}
				}
			}
			else
			{
				if (children())
				{
					m_bb = children()->first.boundingBox();
					m_bb.extendBy(children()->second.boundingBox());
				}
			}

			return m_bb;
		}
		void construct(
			const MeshStructure& structure,
			const objects_iterator begin, const objects_iterator end,
			const uint32_t depth = 0)
		{
			// all object can be stored as a tree leaf
			if (depth > max_depth || std::distance(begin, end) <= leaf_size || 
				(depth == 0 && std::distance(begin, end) <= 32))
			{
				m_objects = objects_t(std::make_move_iterator(begin), std::make_move_iterator(end));
				return;
			}

			// find all objects not suitable for further sub-partition (are too large)
			const auto node_size = boundingBox().max - boundingBox().min;
			auto size_split_point = std::partition(begin, end,
				[&structure, node_size](const auto& object)
				{
					const Math::vec3f object_size = (object->boundingBox(structure).max - object->boundingBox(structure).min);
					return object_size.x < node_size.x&& object_size.y < node_size.y&& object_size.z < node_size.z;
				});
			const auto to_split_count = std::distance(begin, size_split_point);
			const auto too_large_count = std::distance(size_split_point, end);
			if (to_split_count != 0 && too_large_count != 0)
			{
				m_children = std::make_unique<Children>(
					ComponentTreeNode(structure, m_bb, begin, size_split_point, depth + 1), // objects to split
					ComponentTreeNode(structure, m_bb, size_split_point, end, depth + 1), // too large objects
					Children::PartitionType::Size);
				return;
			}
			else if (to_split_count == 0)
			{	// only too large objects left, so further sub-partition is ineffective
				m_objects = objects_t(std::make_move_iterator(size_split_point), std::make_move_iterator(end));
				return;
			}

			const auto to_split_begin = begin;
			const auto to_split_end = size_split_point;
			// find split point
			Math::vec3f split_point{};
			for (int32_t i = 0; i < to_split_count; i++)
				split_point += (to_split_begin[i]->boundingBox(structure).centroid() - split_point) / float(i + 1);

			// count objects and compute distribution variance along each plane
			Math::vec3f variance_sum(0.0f);
			Math::vec3<uint32_t> split_count;
			for (int32_t i = 0; i < to_split_count; i++)
			{
				const auto diff = to_split_begin[i]->boundingBox(structure).centroid() - split_point;
				variance_sum += diff * diff;
				split_count.x += uint32_t(to_split_begin[i]->boundingBox(structure).centroid().x < split_point.x);
				split_count.y += uint32_t(to_split_begin[i]->boundingBox(structure).centroid().y < split_point.y);
				split_count.z += uint32_t(to_split_begin[i]->boundingBox(structure).centroid().z < split_point.z);
			}
			if (split_count == Math::vec3<uint32_t>(0))
			{	// no sub-partition is possible (all objects' centroids are in one single point)
				m_objects = objects_t(std::make_move_iterator(to_split_begin), std::make_move_iterator(to_split_end));
				return;
			}

			// score for each axis
			Math::vec3f	score = variance_sum / float(to_split_count);

			if (score.x >= score.y && score.x >= score.z && split_count.x)
			{	// split by X axis
				auto split_plane = std::partition(to_split_begin, to_split_end,
					[&structure, split_point](const auto& object)
					{ return object->boundingBox(structure).centroid().x < split_point.x; });

				Math::vec3f max = m_bb.max, min = m_bb.min;
				max.x = min.x = split_point.x;
				m_children = std::make_unique<Children>(
					ComponentTreeNode(structure, BoundingBox(m_bb.min, max), to_split_begin, split_plane, depth + 1),
					ComponentTreeNode(structure, BoundingBox(min, m_bb.max), split_plane, to_split_end, depth + 1),
					Children::PartitionType::X);
				return;
			}
			else if (score.y >= score.x && score.y >= score.z && split_count.y)
			{	// split by Y axis
				auto split_plane = std::partition(to_split_begin, to_split_end,
					[&structure, split_point](const auto& object)
					{ return object->boundingBox(structure).centroid().y < split_point.y; });

				Math::vec3f max = m_bb.max, min = m_bb.min;
				max.y = min.y = split_point.y;
				m_children = std::make_unique<Children>(
					ComponentTreeNode(structure, BoundingBox(m_bb.min, max), to_split_begin, split_plane, depth + 1),
					ComponentTreeNode(structure, BoundingBox(min, m_bb.max), split_plane, to_split_end, depth + 1),
					Children::PartitionType::Y);
			}
			else
			{	// split by Z axis
				auto split_plane = std::partition(to_split_begin, to_split_end,
					[&structure, split_point](const auto& object)
					{ return object->boundingBox(structure).centroid().z < split_point.z; });

				Math::vec3f max = m_bb.max, min = m_bb.min;
				max.z = min.z = split_point.z;
				m_children = std::make_unique<Children>(
					ComponentTreeNode(structure, BoundingBox(m_bb.min, max), to_split_begin, split_plane, depth + 1),
					ComponentTreeNode(structure, BoundingBox(min, m_bb.max), split_plane, to_split_end, depth + 1),
					Children::PartitionType::Z);
			}
		}
	};

	template <typename T>
	struct ComponentBVH
	{
	private:
		const MeshStructure& mr_mesh_structure;
		ComponentTreeNode<T> m_root;

	public:
		ComponentBVH(const MeshStructure& structure)
			: mr_mesh_structure(structure)
		{}

	public:
		const ComponentTreeNode<T>& GetRootNode()
		{
			return m_root;
		}

		void construct(ComponentContainer<T>& components)
		{
			std::vector<T*> to_insert;
			for (uint32_t i = 0; i < components.count(); i++)
				to_insert.push_back(&components[i]);
			m_root = ComponentTreeNode<T>{ mr_mesh_structure, std::move(to_insert) };
		}
		void reset()
		{
			m_root.clear();
		}
	};

	template <typename T>
	struct ComponentContainer<T, true>
		: public ComponentContainer<T, false>
	{
	private:
		ComponentBVH<T> m_bvh;

	public:
		ComponentContainer(Updatable* parent,
			const MeshStructure& structure,
			const uint32_t& capacity = 32u)
			: ComponentContainer<T, false>(parent, structure, capacity)
			, m_bvh(structure)
		{}

	public:
		ComponentBVH<T>& getBVH()
		{
			return m_bvh;
		}
		const ComponentBVH<T>& getBVH() const
		{
			return m_bvh;
		}

		void update() override
		{
			m_bvh.construct(*this);
		}
	};
}

#endif
