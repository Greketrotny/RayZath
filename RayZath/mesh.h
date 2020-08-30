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

	class Mesh : public RenderObject
	{
	public:
		typedef Math::vec3<float> Vertex;
		struct VertexStorage
		{
		private:
			Mesh* mesh = nullptr;
			bool* vertexExist = nullptr;
			Vertex* rawVertices = nullptr;
			Vertex* trsVertices = nullptr;
			unsigned int count, capacity;
		public:
			enum VertexType
			{
				VertexTypeRaw,
				VertexTypeTransposed
			};


		private:
			VertexStorage(Mesh* mesh, unsigned int maxVerticesCount);
			~VertexStorage();


		public:
			Vertex* operator[](unsigned int index);
			const Vertex* operator[](unsigned int index) const;


		public:
			Vertex* CreateVertex(float x = 0.0f, float y = 0.0f, float z = 0.0f);
			//bool DestroyVertex(const Vertex* const vertex);
			void DestroyAllVertices(unsigned int newCapacity = 128u);
			template <VertexType T> Vertex* GetVertex(unsigned int index)
			{
				return this->operator[](index);
			}
			template<> Vertex* GetVertex<VertexType::VertexTypeRaw>(unsigned int index)
			{
				return (vertexExist[index]) ? &rawVertices[index] : nullptr;
			}
			template<> Vertex* GetVertex<VertexType::VertexTypeTransposed>(unsigned int index)
			{
				return (vertexExist[index]) ? &trsVertices[index] : nullptr;
			}


		public:
			const unsigned int& Count;
			const unsigned int& Capacity;


		public:
			friend class Mesh;
			friend class CudaMesh;
			friend struct CudaVertexStorage;
		};
		struct TexcrdStorage
		{
			// -- TexcrdStorage::fields -- //
		private:
			Mesh* mesh = nullptr;
			bool* texcdsExist = nullptr;
			Texcrd* texcrds = nullptr;
			unsigned int count, capacity;

			// -- TexcrdStorage::constructor -- //
		private:
			TexcrdStorage(Mesh* mesh, unsigned int maxTexcdsCount);
			~TexcrdStorage();

			// -- TexcrdStorage::operators -- //
		public:
			Texcrd* operator[](unsigned int index) const;

			// -- TexcrdStorage::methods -- //
		public:
			Texcrd* CreateTexcd(float u = 0.0f, float v = 0.0f);
			void DestroyAllTexcds(unsigned int newCapacity = 128u);

			// -- TexcrdStorage::properties -- //
		public:
			const unsigned int& Count;
			const unsigned int& Capacity;

			// -- TexcrdStorage::friends -- //
		public:
			friend class Mesh;
			friend class CudaMesh;
			friend struct CudaTexcrdStorage;
		};
		struct TriangleStorage
		{
			// -- TriangleStorage::fields -- //
		private:
			Mesh* mesh = nullptr;
			bool* triangleExist = nullptr;
			Triangle* rawTriangles = nullptr;
			Triangle* trsTriangles = nullptr;
			unsigned int count, capacity;
		public:
			enum TriangleType
			{
				TriangleTypeRaw,
				TriangleTypeTransposed
			};

			// -- TriangleStorage::constructor -- //
		private:
			TriangleStorage(Mesh* mesh, unsigned int maxTrianglesCount);
			~TriangleStorage();

			// -- TriangleStorage::operators -- //
		public:
			Triangle* operator[](unsigned int index);
			const Triangle* operator[](unsigned int index) const;

			// -- TriangleStorage::methods -- //
		public:
			Triangle* CreateTriangle(
				Vertex* const V1, Vertex* const V2, Vertex* const V3,
				Texcrd* const T1 = nullptr, Texcrd* const T2 = nullptr, Texcrd* const T3 = nullptr,
				Graphics::Color color = Graphics::Color(255u, 255u, 255u));
			bool DestroyTriangle(const Triangle* const triangle);
			void DestroyAllTriangles(unsigned int newCapacity = 128u);
			template <TriangleType T> Triangle* GetTriangle(unsigned int index)
			{
				return this->operator[](index);
			}
			template<> Triangle* GetTriangle<TriangleType::TriangleTypeRaw>(unsigned int index)
			{
				return (triangleExist[index]) ? &rawTriangles[index] : nullptr;
			}
			template<> Triangle* GetTriangle<TriangleType::TriangleTypeTransposed>(unsigned int index)
			{
				return (triangleExist[index]) ? &trsTriangles[index] : nullptr;
			}
			//void SetMaxCount(unsigned int newMaxCount);

			// -- TriangleStorage::properties -- //
		public:
			const unsigned int& Count;
			const unsigned int& Capacity;

			// -- TriangleStorage::friends -- //
		public:
			friend class Mesh;
			friend class CudaMesh;
			friend struct CudaTriangleStorage;
		};
	private:
		VertexStorage m_vertices;
		TexcrdStorage m_texcrds;
		TriangleStorage m_triangles;
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
		void DestroyAllComponents();
		void DestroyAllComponents(unsigned int newVerticesCapacity, unsigned int newTrianglesCapacity);
		void TransposeComponents();
		bool LoadFromFile(std::string file);
		void LoadTexture(const Texture& newTexture);
		void UnloadTexture();

		const Texture* GetTexture() const;
	private:
		void Update() override;


	public:
		VertexStorage& Vertices;
		TriangleStorage& Triangles;
		TexcrdStorage& Texcrds;


	public:
		friend class World;
		friend class CudaMesh;
	};


	template<> struct ConStruct<Mesh> : public ConStruct<RenderObject>
	{
		unsigned int maxVerticesCount, maxTexcrdsCount, maxTrianglesCount;

		ConStruct(
			const ConStruct<RenderObject>& renderObjectConStruct = ConStruct<RenderObject>(),
			unsigned int maxVerticesCount = 128u,
			unsigned int maxTrianglesCount = 128u,
			unsigned int maxTexcrdsCount = 128u)
			: ConStruct<RenderObject>(renderObjectConStruct)
			, maxVerticesCount(maxVerticesCount)
			, maxTrianglesCount(maxTrianglesCount)
			, maxTexcrdsCount(maxTexcrdsCount)
		{}
		~ConStruct()
		{}
	};
}

#endif // !MESH_H