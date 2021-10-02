#ifndef COMPONENT_CONTAINER_H
#define COMPONENT_CONTAINER_H

#include "render_parts.h"
#include "rzexception.h"
#include "mesh_component.h"

#include <algorithm>

namespace RayZath
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
		const uint32_t& GetCount() const
		{
			return m_count;
		}
		const uint32_t& GetCapacity() const
		{
			return m_capacity;
		}
		static uint32_t GetEndPos()
		{
			return sm_npos;
		}

		uint32_t Add(const T& new_component)
		{
			GrowIfNecessary();
			new (&mp_memory[m_count]) T(new_component);

			GetStateRegister().RequestUpdate();
			return m_count++;
		}
		uint32_t Add(T&& new_component)
		{
			GrowIfNecessary();
			new (&mp_memory[m_count]) T(std::move(new_component));

			GetStateRegister().RequestUpdate();
			return m_count++;
		}
		void RemoveAll()
		{
			Resize(0u);
			GetStateRegister().RequestUpdate();
		}
	private:

		// resize capacity if is equal to the current count (memory is full) or 
		// count is below a half of capacity
		void GrowIfNecessary()
		{
			if (m_count >= m_capacity)
				Resize(std::max(uint32_t(m_capacity * 2u), sm_min_capacity));
		}
		void ShrinkIfNecessary()
		{
			if (m_count < m_capacity / 2u)
				Resize(std::max(uint32_t(m_capacity / 2u), sm_min_capacity));
		}
		void Resize(uint32_t capacity)
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

			GetStateRegister().RequestUpdate();
		}
	};

	template <class T>
	struct ComponentTreeNode
	{
	private:
		static constexpr uint32_t s_leaf_size = 8u;
		ComponentTreeNode* m_child[8];
		std::vector<const T*> objects;
		BoundingBox m_bb;
		bool m_is_leaf;


	public:
		ComponentTreeNode(BoundingBox bb = BoundingBox())
			: m_bb(bb)
			, m_is_leaf(true)
		{
			for (int i = 0; i < 8; i++)
				m_child[i] = nullptr;
		}
		~ComponentTreeNode()
		{
			for (int i = 0; i < 8; i++)
			{
				if (m_child[i]) delete m_child[i];
				m_child[i] = nullptr;
			}

			objects.clear();
			m_is_leaf = true;
		}


	public:
		bool Insert(
			const T* object,
			uint32_t depth = 0u)
		{
			if (m_is_leaf)
			{
				if (depth > 15u || objects.size() < s_leaf_size)
				{	// insert the object into leaf

					objects.push_back(object);
				}
				else
				{	// turn leaf into node and reinsert

					m_is_leaf = false;

					// copy objects to temporal storage
					std::vector<const T*> node_objects = objects;
					// add new object to storage
					node_objects.push_back(object);
					objects.clear();

					// distribute objects into child nodes
					for (uint32_t i = 0u; i < node_objects.size(); i++)
					{
						// find child id for the object
						Math::vec3f vCP =
							node_objects[i]->GetBoundingBox().GetCentroid() -
							m_bb.GetCentroid();

						int child_id = 0;
						Math::vec3f child_extent = m_bb.max;
						if (vCP.x < 0.0f)
						{
							child_id += 4;
							child_extent.x = m_bb.min.x;
						}
						if (vCP.y < 0.0f)
						{
							child_id += 2;
							child_extent.y = m_bb.min.y;
						}
						if (vCP.z < 0.0f)
						{
							child_id += 1;
							child_extent.z = m_bb.min.z;
						}

						// insert object to the corresponding child node
						if (!m_child[child_id]) m_child[child_id] = new ComponentTreeNode<T>(BoundingBox(
							m_bb.GetCentroid(), child_extent));
						m_child[child_id]->Insert(node_objects[i], depth + 1);
					}
				}
			}
			else
			{
				// find child id for the object
				Math::vec3f vCP =
					object->GetBoundingBox().GetCentroid() -
					m_bb.GetCentroid();

				int child_id = 0;
				Math::vec3f child_extent = m_bb.max;
				if (vCP.x < 0.0f)
				{
					child_id += 4;
					child_extent.x = m_bb.min.x;
				}
				if (vCP.y < 0.0f)
				{
					child_id += 2;
					child_extent.y = m_bb.min.y;
				}
				if (vCP.z < 0.0f)
				{
					child_id += 1;
					child_extent.z = m_bb.min.z;
				}

				// insert object to the corresponding child node
				if (!m_child[child_id]) m_child[child_id] = new ComponentTreeNode<T>(BoundingBox(
					m_bb.GetCentroid(), child_extent));
				m_child[child_id]->Insert(object, depth + 1u);
			}

			return true;
		}
		bool InsertVector(
			const MeshStructure& structure,
			const std::vector<const T*>& components,
			uint32_t depth)
		{
			Reset();

			if (depth > 15u || components.size() < s_leaf_size || (depth == 0u && components.size() < 32u))
			{
				objects = components;
				return true;
			}
			m_is_leaf = false;

			// ~~~~ On X ~~~~ //
			// find x_plane position
			float x_plane = this->m_bb.GetCentroid().x;
			for (size_t i = 0u; i < components.size(); i++)
			{
				x_plane += (components[i]->GetBoundingBox(structure).GetCentroid().x - x_plane) / float(i + 1u);
			}

			// split components
			std::vector<const T*> x_split[2];
			for (size_t i = 0u; i < components.size(); i++)
			{
				if (components[i]->GetBoundingBox(structure).GetCentroid().x > x_plane)
				{
					x_split[0].push_back(components[i]);
				}
				else
				{
					x_split[1].push_back(components[i]);
				}
			}


			// ~~~~ On Y ~~~~ //
			// find y_planes position
			float y_plane[2];
			y_plane[0] = this->m_bb.GetCentroid().y;
			y_plane[1] = y_plane[0];
			for (size_t x = 0u; x < 2u; x++)
			{
				for (size_t i = 0u; i < x_split[x].size(); i++)
				{
					y_plane[x] += (x_split[x][i]->GetBoundingBox(structure).GetCentroid().y - y_plane[x]) / float(i + 1u);
				}
			}

			// split components
			std::vector<const T*> y_split[4];
			for (size_t x = 0u; x < 2u; x++)
			{
				for (size_t i = 0u; i < x_split[x].size(); i++)
				{
					if (x_split[x][i]->GetBoundingBox(structure).GetCentroid().y > y_plane[x])
					{
						y_split[2u * x + 0u].push_back(x_split[x][i]);
					}
					else
					{
						y_split[2u * x + 1u].push_back(x_split[x][i]);
					}
				}
			}


			// ~~~~ On Z ~~~~ //
			// find z_planes position
			float z_plane[4];
			for (int i = 0; i < 4; i++) z_plane[i] = this->m_bb.GetCentroid().z;
			for (size_t y = 0u; y < 4u; y++)
			{
				for (size_t i = 0u; i < y_split[y].size(); i++)
				{
					z_plane[y] += (y_split[y][i]->GetBoundingBox(structure).GetCentroid().z - z_plane[y]) / float(i + 1u);
				}
			}

			// split components
			std::vector<const T*> z_split[8];
			for (size_t y = 0u; y < 4u; y++)
			{
				for (size_t i = 0u; i < y_split[y].size(); i++)
				{
					if (y_split[y][i]->GetBoundingBox(structure).GetCentroid().z > z_plane[y])
					{
						z_split[2u * y + 0u].push_back(y_split[y][i]);
					}
					else
					{
						z_split[2u * y + 1u].push_back(y_split[y][i]);
					}
				}
			}

			// insert object to the corresponding child node
			for (uint8_t i = 0u; i < 8u; i++)
			{
				Math::vec3f parent_extent = m_bb.max;
				if ((i >> 2u) & 0x1) parent_extent.x = m_bb.min.x;
				if ((i >> 1u) & 0x1) parent_extent.y = m_bb.min.y;
				if ((i >> 0u) & 0x1) parent_extent.z = m_bb.min.z;

				if (z_split[i].size() > 0u)
				{
					if (!m_child[i]) m_child[i] = new ComponentTreeNode<T>(BoundingBox(
						parent_extent, Math::vec3f(x_plane, y_plane[(i >> 1u) & 0x1], z_plane[(i >> 0u) & 0x1])));
					m_child[i]->InsertVector(structure, z_split[i], depth + 1u);
				}
			}

			return true;
		}
		bool InsertVectorSorted(
			const std::vector<const T*>& components,
			uint32_t depth = 0u)
		{
			Reset();

			if (depth > 15u || components.size() < s_leaf_size)
			{
				objects = components;
				return true;
			}
			m_is_leaf = false;

			// ~~~~ Along X ~~~~ //
			std::vector<const T*> sorted_x = components;
			std::sort(sorted_x.begin(), sorted_x.end(), [](const T* left, const T* right) {
				return left->GetBoundingBox().GetCentroid().x < right->GetBoundingBox().GetCentroid().x;
				});

			const float x_plane = sorted_x[sorted_x.size() / 2u]->GetBoundingBox().GetCentroid().x;

			// split components
			std::vector<const T*> x_split[2];
			x_split[0] = std::vector<const T*>(sorted_x.begin() + sorted_x.size() / 2u, sorted_x.end());
			x_split[1] = std::vector<const T*>(sorted_x.begin(), sorted_x.begin() + sorted_x.size() / 2u);


			// ~~~~ Along Y ~~~~ //
			// find y_planes position
			float y_plane[2];
			for (int i = 0; i < 2; i++)
			{
				std::sort(x_split[i].begin(), x_split[i].end(), [](const T* left, const T* right) {
					return left->GetBoundingBox().GetCentroid().y < right->GetBoundingBox().GetCentroid().y;
					});
				y_plane[i] = x_split[i][x_split->size() / 2u]->GetBoundingBox().GetCentroid().y;
			}

			// split components
			std::vector<const T*> y_split[4];
			for (uint8_t x = 0u; x < 2u; x++)
			{
				y_split[2u * x + 0u] = std::vector<const T*>(x_split[x].begin() + x_split[x].size() / 2u, x_split[x].end());
				y_split[2u * x + 1u] = std::vector<const T*>(x_split[x].begin(), x_split[x].begin() + x_split[x].size() / 2u);
			}


			// ~~~~ Along Z ~~~~ //
			// find z_planes position
			float z_plane[4];
			for (int i = 0; i < 4; i++)
			{
				std::sort(y_split[i].begin(), y_split[i].end(), [](const T* left, const T* right) {
					return left->GetBoundingBox().GetCentroid().z < right->GetBoundingBox().GetCentroid().z;
					});
				z_plane[i] = y_split[i][y_split->size() / 2u]->GetBoundingBox().GetCentroid().z;
			}

			// split components
			std::vector<const T*> z_split[8];
			for (uint8_t y = 0u; y < 4u; y++)
			{
				z_split[2u * y + 0u] = std::vector<const T*>(y_split[y].begin() + y_split[y].size() / 2u, y_split[y].end());
				z_split[2u * y + 1u] = std::vector<const T*>(y_split[y].begin(), y_split[y].begin() + y_split[y].size() / 2u);
			}


			// insert object to the corresponding child node
			for (uint8_t i = 0u; i < 8u; i++)
			{
				Math::vec3f parent_extent = m_bb.max;
				if ((i >> 2u) & 0x1) parent_extent.x = m_bb.min.x;
				if ((i >> 1u) & 0x1) parent_extent.y = m_bb.min.y;
				if ((i >> 0u) & 0x1) parent_extent.z = m_bb.min.z;

				if (z_split[i].size() > 0u)
				{
					if (!m_child[i]) m_child[i] = new ComponentTreeNode<T>(BoundingBox(
						parent_extent, Math::vec3f(x_plane, y_plane[(i >> 1u) & 0x1], z_plane[(i >> 0u) & 0x1])));
					m_child[i]->InsertVectorSorted(z_split[i], depth + 1u);
				}
			}

			return true;
		}
		bool Remove(const T* object)
		{
			if (m_is_leaf)
			{
				for (uint32_t i = 0; i < objects.size(); ++i)
				{
					if (object == objects[i])
					{
						objects.erase(objects.begin() + i);
						return true;
					}
				}
			}
			else
			{
				for (int i = 0; i < 8; i++)
				{
					if (m_child[i])
					{
						if (m_child[i].Remove(object))
							return true;
					}
				}
			}

			return false;
		}
		BoundingBox FitBoundingBox(const MeshStructure& structure)
		{
			if (objects.size() > 0u)
			{
				m_bb = objects[0]->GetBoundingBox(structure);
				for (auto* o : objects)
				{
					m_bb.ExtendBy(o->GetBoundingBox(structure));
				}
				return m_bb;
			}
			else
			{
				int i = 0;

				while (i < 8)
				{
					if (m_child[i])
					{
						m_bb = m_child[i]->FitBoundingBox(structure);
						i++;
						break;
					}
					i++;
				}
				while (i < 8)
				{
					if (m_child[i])
						m_bb.ExtendBy(m_child[i]->FitBoundingBox(structure));

					i++;
				}
				return m_bb;
			}
		}
		void Reset()
		{
			for (int i = 0; i < 8; i++)
			{
				if (m_child[i]) delete m_child[i];
				m_child[i] = nullptr;
			}

			objects.clear();
			m_is_leaf = true;
		}

		void SetBoundingBox(const BoundingBox& bb)
		{
			m_bb = bb;
		}
		void ExtendBoundingBox(const BoundingBox& bb)
		{
			m_bb.ExtendBy(bb);
		}

		ComponentTreeNode* GetChild(unsigned int child_id)
		{
			return m_child[child_id];
		}
		const ComponentTreeNode* GetChild(unsigned int child_id) const
		{
			return m_child[child_id];
		}
		uint32_t GetChildCount() const
		{
			uint32_t child_count = 0u;
			for (uint32_t i = 0u; i < 8u; i++)
				if (m_child[i] != nullptr)
					child_count++;

			return child_count;
		}
		uint32_t GetRecursiveChildCount() const
		{
			uint32_t child_count = 0u;
			for (int i = 0; i < 8; i++)
			{
				if (m_child[i])
				{
					child_count += m_child[i]->GetRecursiveChildCount() + 1u;
				}
			}
			return child_count;
		}

		const T* GetObject(unsigned int object_index) const
		{
			return objects[object_index];
		}
		uint32_t GetObjectCount() const
		{
			return uint32_t(objects.size());
		}

		BoundingBox GetBoundingBox() const
		{
			return m_bb;
		}

		bool IsLeaf() const
		{
			return m_is_leaf;
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
		void Construct(const ComponentContainer<T>& components)
		{
			Reset();

			if (components.GetCount() == 0u) return;

			// Expand root BB by BBs of all components
			m_root.SetBoundingBox(components[0].GetBoundingBox(mr_mesh_structure));
			for (uint32_t i = 1u; i < components.GetCount(); i++)
			{
				m_root.ExtendBoundingBox(components[i].GetBoundingBox(mr_mesh_structure));
			}

			// Insert all components into BVH (sorted vector)
			/*std::vector<const T*> com_ps;
			for (uint32_t i = 0u; i < components.GetCount(); i++)
			{
				com_ps.push_back(&components[i]);
			}
			m_root.InsertVectorSorted(com_ps);*/

			// Insert all components into BVH (vector)
			std::vector<const T*> com_ps;
			for (uint32_t i = 0u; i < components.GetCount(); i++)
			{
				com_ps.push_back(&components[i]);
			}
			m_root.InsertVector(mr_mesh_structure, com_ps, 0u);

			// Insert all components into BVH (sequentially)
			/*for (uint32_t i = 0u; i < components.GetCount(); i++)
			{
				m_root.Insert(&components[i]);
			}*/


			// Fit bounding boxes of each tree node
			m_root.FitBoundingBox(mr_mesh_structure);
		}
		void Reset()
		{
			m_root.Reset();
		}

		const ComponentTreeNode<T>& GetRootNode()
		{
			return m_root;
		}
		uint32_t GetTreeSize()
		{
			return 1u + m_root.GetRecursiveChildCount();
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
		ComponentBVH<T>& GetBVH()
		{
			return m_bvh;
		}
		const ComponentBVH<T>& GetBVH() const
		{
			return m_bvh;
		}

		void Update() override
		{
			m_bvh.Construct(*this);
		}
	};
}

#endif
