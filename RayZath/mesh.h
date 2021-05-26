#ifndef MESH_H
#define MESH_H

#include "render_object.h"
#include "rzexception.h"

#include "bitmap.h"
#include "color.h"

#include <vector>
#include <algorithm>

namespace RayZath
{
	class Mesh;
	template<> struct ConStruct<Mesh>;

	template <typename T, bool WithBVH = std::is_same<T, Triangle>::value>
	struct ComponentContainer;

	template <typename T>
	struct ComponentContainer<T, false>
		: public Updatable
	{
	private:
		T* mp_memory;
		uint32_t m_capacity, m_count;


	public:
		ComponentContainer(
			Updatable* parent,
			const uint32_t& capacity = 32u)
			: Updatable(parent)
			, m_capacity(capacity)
			, m_count(0u)
		{
			mp_memory = (T*)std::malloc(m_capacity * sizeof(T));
		}
		~ComponentContainer()
		{
			if (mp_memory) std::free(mp_memory);
			mp_memory = nullptr;
			m_capacity = 0u;
			m_count = 0u;
		}


	public:
		T& operator[](const uint32_t index)
		{
			return mp_memory[index];
		}
		const T& operator[](const uint32_t index) const
		{
			return mp_memory[index];
		}


	protected:
		void Resize(uint32_t capacity)
		{
			if (m_capacity == capacity) return;
			capacity = std::max(capacity, 2u);

			T* mp_new_memory = (T*)std::malloc(capacity * sizeof(T));
			RZAssert(mp_new_memory != nullptr, "malloc returned nullptr");
			std::memcpy(mp_new_memory, mp_memory, std::min(m_capacity, capacity) * sizeof(T));

			if (mp_memory) std::free(mp_memory);
			mp_memory = mp_new_memory;

			m_capacity = capacity;
			m_count = std::min(m_count, m_capacity);

			GetStateRegister().RequestUpdate();
		}
		void Reset(uint32_t capacity = 0u)
		{
			Resize(capacity);
			m_count = 0u;
			GetStateRegister().RequestUpdate();
		}
		T* Add(const T& new_object)
		{
			if (m_count >= m_capacity)
				Resize(std::max(uint32_t(m_capacity * 1.5f), m_capacity + 2u));
			new (&mp_memory[m_count++]) T(new_object);

			GetStateRegister().RequestUpdate();
			return mp_memory + m_count - 1u;
		}
	public:
		const uint32_t& GetCapacity() const
		{
			return m_capacity;
		}
		const uint32_t& GetCount() const
		{
			return m_count;
		}

		friend struct MeshStructure;
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
				x_plane += (components[i]->GetBoundingBox().GetCentroid().x - x_plane) / float(i + 1u);
			}

			// split components
			std::vector<const T*> x_split[2];
			for (size_t i = 0u; i < components.size(); i++)
			{
				if (components[i]->GetBoundingBox().GetCentroid().x > x_plane)
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
					y_plane[x] += (x_split[x][i]->GetBoundingBox().GetCentroid().y - y_plane[x]) / float(i + 1u);
				}
			}

			// split components
			std::vector<const T*> y_split[4];
			for (size_t x = 0u; x < 2u; x++)
			{
				for (size_t i = 0u; i < x_split[x].size(); i++)
				{
					if (x_split[x][i]->GetBoundingBox().GetCentroid().y > y_plane[x])
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
					z_plane[y] += (y_split[y][i]->GetBoundingBox().GetCentroid().z - z_plane[y]) / float(i + 1u);
				}
			}

			// split components
			std::vector<const T*> z_split[8];
			for (size_t y = 0u; y < 4u; y++)
			{
				for (size_t i = 0u; i < y_split[y].size(); i++)
				{
					if (y_split[y][i]->GetBoundingBox().GetCentroid().z > z_plane[y])
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
					m_child[i]->InsertVector(z_split[i], depth + 1u);
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
		BoundingBox FitBoundingBox()
		{
			if (objects.size() > 0u)
			{
				m_bb = objects[0]->GetBoundingBox();
				for (auto* o : objects)
				{
					m_bb.ExtendBy(o->GetBoundingBox());
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
						m_bb = m_child[i]->FitBoundingBox();
						i++;
						break;
					}
					i++;
				}
				while (i < 8)
				{
					if (m_child[i])
						m_bb.ExtendBy(m_child[i]->FitBoundingBox());

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
			for (int i = 0; i < 8; i++)
			{
				if (m_child[i])
				{
					child_count += m_child[i]->GetChildCount() + 1u;
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
		ComponentTreeNode<T> m_root;


	public:
		ComponentBVH()
		{}
		~ComponentBVH()
		{}


	public:
		void Construct(const ComponentContainer<T>& components)
		{
			Reset();

			if (components.GetCount() == 0u) return;

			// Expand root BB by BBs of all components
			m_root.SetBoundingBox(components[0].GetBoundingBox());
			for (uint32_t i = 1u; i < components.GetCount(); i++)
			{
				m_root.ExtendBoundingBox(components[i].GetBoundingBox());
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
			m_root.InsertVector(com_ps, 0u);

			// Insert all components into BVH (sequentially)
			/*for (uint32_t i = 0u; i < components.GetCount(); i++)
			{
				m_root.Insert(&components[i]);
			}*/


			// Fit bounding boxes of each tree node
			m_root.FitBoundingBox();
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
			return 1u + m_root.GetChildCount();
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
			const uint32_t& capacity = 32u)			
			: ComponentContainer<T, false>(parent, capacity)
		{

		}
		~ComponentContainer()
		{

		}


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


		friend struct MeshStructure;
	};


	struct MeshStructure
		: public Updatable
	{
	private:
		ComponentContainer<Math::vec3f> m_vertices;
		ComponentContainer<Texcrd> m_texcrds;
		ComponentContainer<Math::vec3f> m_normals;
		ComponentContainer<Triangle> m_triangles;


	public:
		MeshStructure(
			Updatable* updatable,
			const ConStruct<MeshStructure>& conStruct);
		~MeshStructure();


	public:
		bool LoadFromFile(const std::wstring& file_name);

		Vertex* CreateVertex(const Math::vec3f& vertex);
		Vertex* CreateVertex(const float& x, const float& y, const float& z);

		Texcrd* CreateTexcrd(const Texcrd& texcrd);
		Texcrd* CreateTexcrd(const float& u, const float& v);

		Math::vec3f* CreateNormal(const Math::vec3f& normal);
		Math::vec3f* CreateNormal(const float& x, const float& y, const float& z);

		bool CreateTriangle(
			const uint32_t& v1, const uint32_t& v2, const uint32_t& v3,
			const uint32_t& t1, const uint32_t& t2, const uint32_t& t3,
			const uint32_t& n1, const uint32_t& n2, const uint32_t& n3,
			const uint32_t& material_id);
		bool CreateTriangle(
			Vertex* v1, Vertex* v2, Vertex* v3,
			Texcrd* t1 = nullptr, Texcrd* t2 = nullptr, Texcrd* t3 = nullptr,
			Normal* n1 = nullptr, Normal* n2 = nullptr, Normal* n3 = nullptr,
			const uint32_t& material_id = 0u);

		void Reset();
		void Reset(
			const uint32_t& vertices_capacity,
			const uint32_t& texcrds_capacity,
			const uint32_t& normals_capacity,
			const uint32_t& triangles_capacity);

		ComponentContainer<Math::vec3f>& GetVertices();
		ComponentContainer<Texcrd>& GetTexcrds();
		ComponentContainer<Math::vec3f>& GetNormals();
		ComponentContainer<Triangle>& GetTriangles();
		const ComponentContainer<Math::vec3f>& GetVertices() const;
		const ComponentContainer<Texcrd>& GetTexcrds() const;
		const ComponentContainer<Math::vec3f>& GetNormals() const;
		const ComponentContainer<Triangle>& GetTriangles() const;

		void Update() override;
	};	
	template <> struct ConStruct<MeshStructure>
	{
		uint32_t vertices, texcrds, normals, triangles;

		ConStruct(
			const uint32_t& vertices = 2u,
			const uint32_t& texcrds = 2u,
			const uint32_t& normals = 2u,
			const uint32_t& triangles = 2u)
			: vertices(vertices)
			, texcrds(texcrds)
			, normals(normals)
			, triangles(triangles)
		{}
	};


	class Mesh : public RenderObject
	{
		static constexpr uint32_t sm_mat_count = 64u;
	private:
		Observer<MeshStructure> m_mesh_structure;
		Observer<Material> m_materials[sm_mat_count];


	public:
		Mesh(const Mesh&) = delete;
		Mesh(Mesh&&) = delete;
		Mesh(
			Updatable* updatable,
			const ConStruct<Mesh>& conStruct);
		~Mesh();


	public:
		Mesh& operator=(const Mesh&) = delete;
		Mesh& operator=(Mesh&&) = delete;


	public:
		void SetMeshStructure(const Handle<MeshStructure>& mesh_structure);
		void SetMaterial(
			const Handle<Material>& material,
			const uint32_t& material_index);

		const Handle<MeshStructure>& GetMeshStructure() const;
		const Handle<Material>& GetMaterial(const uint32_t& material_index) const;
		static constexpr uint32_t GetMaterialCount()
		{
			return sm_mat_count;
		}
	public:
		void Update() override;
		void NotifyMeshStructure();
		void NotifyMaterial();
	private:
		void CalculateBoundingBox();
	};


	template<> struct ConStruct<Mesh> : public ConStruct<RenderObject>
	{
		Handle<MeshStructure> mesh_structure;
		Handle<Material> material;

		ConStruct(
			const std::wstring& name = L"name",
			const Math::vec3f& position = Math::vec3f(0.0f, 0.0f, 0.0f),
			const Math::vec3f& rotation = Math::vec3f(0.0f, 0.0f, 0.0f),
			const Math::vec3f& center = Math::vec3f(0.0f, 0.0f, 0.0f),
			const Math::vec3f& scale = Math::vec3f(1.0f, 1.0f, 1.0f),
			const Handle<MeshStructure>& mesh_structure = Handle<MeshStructure>(),
			const Handle<Material>& material = Handle<Material>())
			: ConStruct<RenderObject>(name, position, rotation, center, scale)
			, mesh_structure(mesh_structure)
			, material(material)
		{}
	};
}

#endif // !MESH_H