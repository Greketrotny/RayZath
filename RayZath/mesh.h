#ifndef MESH_H
#define MESH_H

#include "render_object.h"
#include <vector>

#include "bitmap.h"
#include "color.h"

namespace RayZath
{
	class Mesh;
	template<> struct ConStruct<Mesh>;

	template <typename T>
	struct ComponentStorage
	{
	private:
		T* mp_memory;
		uint32_t m_capacity, m_count;


	public:
		ComponentStorage(const uint32_t& capacity = 32u)
			: m_capacity(capacity)
			, m_count(0u)
		{
			mp_memory = (T*)malloc(m_capacity * sizeof(T));
		}
		~ComponentStorage()
		{
			if (mp_memory) free(mp_memory);
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


	private:
		void Resize(const uint32_t& capacity)
		{
			if (m_capacity == capacity) return;

			T* mp_new_memory = (T*)malloc(capacity * sizeof(T));
			memcpy(mp_new_memory, mp_memory, std::min(m_capacity, capacity) * sizeof(T));

			if (mp_memory) free(mp_memory);
			mp_memory = mp_new_memory;

			m_capacity = capacity;
			m_count = std::min(m_count, m_capacity);
		}
		void Reset()
		{
			m_count = 0u;
		}
		T* Add(const T& new_object)
		{
			if (m_count >= m_capacity) return nullptr;
			new (&mp_memory[m_count++]) T(new_object);
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

		friend struct MeshData;
	};

	struct MeshData
	{
	private:
		ComponentStorage<Math::vec3<float>> m_vertices;
		ComponentStorage<Texcrd> m_texcrds;
		// ComponentStorage<Math::vec3<float>> m_normals;
		ComponentStorage<Triangle> m_triangles;


	public:
		MeshData();
		MeshData(
			const uint32_t& vertices, 
			const uint32_t& texcrds, 
			const uint32_t& triangles);
		~MeshData();


	public:
		//bool LoadFromFile(const std::wstring& file_name);

		Vertex* CreateVertex(const Math::vec3<float>& vertex);
		Vertex* CreateVertex(const float& x, const float& y, const float& z);

		Texcrd* CreateTexcrd(const Texcrd& texcrd);
		Texcrd* CreateTexcrd(const float& u, const float& v);

		bool CreateTriangle(
			const uint32_t& v1, const uint32_t& v2, const uint32_t& v3,
			const uint32_t& t1, const uint32_t& t2, const uint32_t& t3,
			const Graphics::Color& color = Graphics::Color(0xFF, 0xFF, 0xFF, 0x00));
		bool CreateTriangle(
			Vertex* v1, Vertex* v2, Vertex* v3,
			Texcrd* t1 = nullptr, Texcrd* t2 = nullptr, Texcrd* t3 = nullptr,
			const Graphics::Color& color = Graphics::Color(0xFF, 0xFF, 0xFF, 0x00));

		void Reset();
		void Reset(
			const uint32_t& vertices_capacity,
			const uint32_t& texcrds_capacity,
			const uint32_t& triangles_capacity);

		ComponentStorage<Math::vec3<float>>& GetVertices();
		ComponentStorage<Texcrd>& GetTexcrds();
		ComponentStorage<Triangle>& GetTriangles();
		const ComponentStorage<Math::vec3<float>>& GetVertices() const;
		const ComponentStorage<Texcrd>& GetTexcrds() const;
		const ComponentStorage<Triangle>& GetTriangles() const;
	};

	class Mesh : public RenderObject
	{
	private:
		MeshData m_mesh_data;
		Texture* m_pTexture = nullptr;


	private:
		Mesh(const Mesh&) = delete;
		Mesh(Mesh&&) = delete;
		Mesh(
			const size_t& id,
			Updatable* updatable,
			const ConStruct<Mesh>& conStruct);
		~Mesh();


	public:
		Mesh& operator=(const Mesh&) = delete;
		Mesh& operator=(Mesh&&) = delete;


	public:
		void LoadTexture(const Texture& newTexture);
		void UnloadTexture();
		void TransposeComponents();

		const Texture* GetTexture() const;
		MeshData& GetMeshData();
		const MeshData& GetMeshData() const;
	public:
		void Update() override;


	public:
		friend class ObjectCreator;
		friend class CudaMesh;
	};


	template<> struct ConStruct<Mesh> : public ConStruct<RenderObject>
	{
		uint32_t vertices_capacity, texcrds_capacity, triangles_capacity;

		ConStruct(
			const ConStruct<RenderObject>& renderObjectConStruct = ConStruct<RenderObject>(),
			unsigned int vertices_capacity = 128u,
			unsigned int texcrds_capacity = 128u,
			unsigned int triangles_capacity = 128u)
			: ConStruct<RenderObject>(renderObjectConStruct)
			, vertices_capacity(vertices_capacity)
			, texcrds_capacity(texcrds_capacity)
			, triangles_capacity(triangles_capacity)
		{}
		~ConStruct()
		{}
	};
}

#endif // !MESH_H