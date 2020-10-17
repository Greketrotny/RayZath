#ifndef MESH_H
#define MESH_H

#include "render_object.h"

#include "bitmap.h"
#include "color.h"

#include <vector>

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
			mp_memory = (T*)malloc(m_capacity * sizeof(T));
		}
		~ComponentContainer()
		{
			if (mp_memory) free(mp_memory);
			mp_memory = nullptr;
			m_capacity = 0u;
			m_count = 0u;
		}


	public:
		T& operator[](uint32_t index)
		{
			return mp_memory[index];
		}
		const T& operator[](uint32_t index) const
		{
			return mp_memory[index];
		}


	protected:
		void Resize(uint32_t capacity)
		{
			if (m_capacity == capacity) return;
			capacity = std::max(capacity, 2u);

			T* mp_new_memory = (T*)malloc(capacity * sizeof(T));
			memcpy(mp_new_memory, mp_memory, std::min(m_capacity, capacity) * sizeof(T));

			if (mp_memory) free(mp_memory);
			mp_memory = mp_new_memory;

			m_capacity = capacity;
			m_count = std::min(m_count, m_capacity);

			GetStateRegister().RequestUpdate();
		}
		void Reset()
		{
			m_count = 0u;
			GetStateRegister().RequestUpdate();
		}
		T* Add(const T& new_object)
		{
			if (m_count >= m_capacity) return nullptr;
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
						Math::vec3<float> vCP =
							node_objects[i]->GetBoundingBox().GetCentroid() -
							m_bb.GetCentroid();

						int child_id = 0;
						Math::vec3<float> child_extent = m_bb.max;
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
				Math::vec3<float> vCP =
					object->GetBoundingBox().GetCentroid() -
					m_bb.GetCentroid();

				int child_id = 0;
				Math::vec3<float> child_extent = m_bb.max;
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
				m_child[child_id]->Insert(object, depth + 1);
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
			for (uint32_t i = 1; i < components.GetCount(); i++)
			{
				m_root.ExtendBoundingBox(components[i].GetBoundingBox());
			}

			// Insert all components into BVH
			for (uint32_t i = 0; i < components.GetCount(); i++)
			{
				m_root.Insert(&components[i]);
			}

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
		ComponentContainer<Math::vec3<float>> m_vertices;
		ComponentContainer<Texcrd> m_texcrds;
		ComponentContainer<Math::vec3<float>> m_normals;
		ComponentContainer<Triangle> m_triangles;


	public:
		MeshStructure(
			Updatable* parent,
			const uint32_t& vertices = 2u, 
			const uint32_t& texcrds = 2u,
			const uint32_t& normals = 2u,
			const uint32_t& triangles = 2u);
		~MeshStructure();


	public:
		bool LoadFromFile(const std::wstring& file_name);

		Vertex* CreateVertex(const Math::vec3<float>& vertex);
		Vertex* CreateVertex(const float& x, const float& y, const float& z);

		Texcrd* CreateTexcrd(const Texcrd& texcrd);
		Texcrd* CreateTexcrd(const float& u, const float& v);

		Math::vec3<float>* CreateNormal(const Math::vec3<float>& normal);
		Math::vec3<float>* CreateNormal(const float& x, const float& y, const float& z);

		bool CreateTriangle(
			const uint32_t& v1, const uint32_t& v2, const uint32_t& v3,
			const uint32_t& t1, const uint32_t& t2, const uint32_t& t3,
			const uint32_t& n1, const uint32_t& n2, const uint32_t& n3,
			const Graphics::Color& color = Graphics::Color(0xFF, 0xFF, 0xFF, 0x00));
		bool CreateTriangle(
			Vertex* v1, Vertex* v2, Vertex* v3,
			Texcrd* t1 = nullptr, Texcrd* t2 = nullptr, Texcrd* t3 = nullptr,
			Normal* n1 = nullptr, Normal* n2 = nullptr, Normal* n3 = nullptr,
			const Graphics::Color& color = Graphics::Color(0xFF, 0xFF, 0xFF, 0x00));

		void Reset();
		void Reset(
			const uint32_t& vertices_capacity,
			const uint32_t& texcrds_capacity,
			const uint32_t& normals_capacity,
			const uint32_t& triangles_capacity);

		ComponentContainer<Math::vec3<float>>& GetVertices();
		ComponentContainer<Texcrd>& GetTexcrds();
		ComponentContainer<Math::vec3<float>>& GetNormals();
		ComponentContainer<Triangle>& GetTriangles();
		const ComponentContainer<Math::vec3<float>>& GetVertices() const;
		const ComponentContainer<Texcrd>& GetTexcrds() const;
		const ComponentContainer<Math::vec3<float>>& GetNormals() const;
		const ComponentContainer<Triangle>& GetTriangles() const;

		void Update() override;
	};

	class Mesh : public RenderObject
	{
	private:
		MeshStructure m_mesh_structure;
		Texture* m_pTexture = nullptr;


	private:
		Mesh(const Mesh&) = delete;
		Mesh(Mesh&&) = delete;
		Mesh(
			const uint32_t& id,
			Updatable* updatable,
			const ConStruct<Mesh>& conStruct);
		~Mesh();


	public:
		Mesh& operator=(const Mesh&) = delete;
		Mesh& operator=(Mesh&&) = delete;


	public:
		void LoadTexture(const Texture& newTexture);
		void UnloadTexture();

		const Texture* GetTexture() const;
		MeshStructure& GetMeshStructure();
		const MeshStructure& GetMeshStructure() const;
	public:
		void Update() override;
	private:
		void CalculateBoundingBox();


	public:
		friend class ObjectCreator;
		friend class CudaMesh;
	};


	template<> struct ConStruct<Mesh> : public ConStruct<RenderObject>
	{
		uint32_t vertices_capacity, texcrds_capacity, normals_capacity, triangles_capacity;

		ConStruct(
			const ConStruct<RenderObject>& renderObjectConStruct = ConStruct<RenderObject>(),
			uint32_t vertices_capacity = 128u,
			uint32_t texcrds_capacity = 128u,
			uint32_t normals_capacity = 128,
			uint32_t triangles_capacity = 128u)
			: ConStruct<RenderObject>(renderObjectConStruct)
			, vertices_capacity(vertices_capacity)
			, texcrds_capacity(texcrds_capacity)
			, normals_capacity(normals_capacity)
			, triangles_capacity(triangles_capacity)
		{}
		~ConStruct()
		{}
	};
}

#endif // !MESH_H