#include "mesh.h"

#include <fstream>
#include <algorithm>

namespace RayZath
{
	// ~~~~~~~~ [CLASS] Mesh ~~~~~~~~
	// -- Mesh::fields -- //
	// >> [STRUCT] Mesh::VertexStorage ---------------|
	// -- // -- VertexStorage::constructor -- //
	Mesh::VertexStorage::VertexStorage(Mesh* mesh, unsigned int maxVerticesCount)
		:mesh(mesh),
		count(0u),
		capacity(maxVerticesCount),
		Count(count),
		Capacity(capacity)
	{
		vertexExist = (bool*)malloc(capacity * sizeof(bool));
		for (unsigned int i = 0u; i < capacity; ++i) vertexExist[i] = false;
		rawVertices = (Vertex*)malloc(capacity * sizeof(Vertex));
		trsVertices = (Vertex*)malloc(capacity * sizeof(Vertex));
	}
	Mesh::VertexStorage::~VertexStorage()
	{
		for (unsigned int i = 0u; i < capacity; ++i)
		{
			if (vertexExist[i])
			{
				rawVertices[i].~vec3();
				trsVertices[i].~vec3();
			}
		}
		free(vertexExist);
		free(rawVertices);
		free(trsVertices);
	}
	// -- // -- VertexStorage::operator -- //
	Mesh::Vertex* Mesh::VertexStorage::operator[](unsigned int index)
	{
		return (vertexExist[index]) ? &rawVertices[index] : nullptr;
	}
	const Mesh::Vertex* Mesh::VertexStorage::operator[](unsigned int index) const
	{
		return (vertexExist[index]) ? &rawVertices[index] : nullptr;
	}
	// -- // -- VertexStorage::methods -- //
	Mesh::Vertex* Mesh::VertexStorage::CreateVertex(float x, float y, float z)
	{
		if (count == capacity)
			return nullptr;

		for (unsigned int i = 0u; i < capacity; ++i)
		{
			if (!vertexExist[i])
			{
				new (&rawVertices[i]) Vertex(x, y, z);
				new (&trsVertices[i]) Vertex(x, y, z);
				vertexExist[i] = true;
				++count;
				return &rawVertices[i];
			}
		}
		return nullptr;
	}
	/*bool Mesh::VertexStorage::DestroyVertex(const Vertex* const vertex)
	{
		if (vertex == nullptr)
			return false;

		for (unsigned int i = 0u; i < maxCount; ++i)
		{
			if (vertex == &rawVertices[i])
			{
				rawVertices[i].~vec3d();
				trsVertices[i].~vec3d();
				vertexExist[i] = false;
				--currCount;

				// TODO: destroy all triangles which point to just deleted vertices


				return true;
			}
		}
		return false;
	}*/
	void Mesh::VertexStorage::DestroyAllVertices(unsigned int newCapacity)
	{
		// delete all vertices 
		for (unsigned int i = 0u; i < capacity; ++i)
		{
			if (vertexExist[i])
			{
				rawVertices[i].~vec3();
				trsVertices[i].~vec3();
			}
			vertexExist[i] = false;
		}

		if (newCapacity == capacity)
			return;

		// free up vertices' memory
		free(rawVertices);
		free(trsVertices);
		free(vertexExist);

		// allocate new sized memory
		capacity = newCapacity;
		count = 0u;
		rawVertices = (Vertex*)malloc(capacity * sizeof(Vertex));
		trsVertices = (Vertex*)malloc(capacity * sizeof(Vertex));
		vertexExist = (bool*)malloc(capacity * sizeof(bool));
		for (unsigned int i = 0u; i < capacity; ++i) vertexExist[i] = false;
	}
	/*void Mesh::VertexStorage::SetMaxCount(unsigned int newMaxCount)
	{
		if (newMaxCount == maxCount)
			return;

		if (newMaxCount > maxCount)
		{
			maxCount = newMaxCount;

			// size vertexExist array up
			vertexExist = (bool*)realloc(vertexExist, maxCount * sizeof(bool));

			// size rawVertices array up
			rawVertices = (Vertex*)realloc(rawVertices, maxCount * sizeof(Vertex));

			// size trsVertices array up
			trsVertices = (Vertex*)realloc(trsVertices, maxCount * sizeof(Vertex));
		}
		else
		{
			// TODO: delete all triangles pointing to vertices to delete
			//for (unsigned int i = 0u; i < mesh->triangles.maxCount; ++i)
			//{
			//	if (mesh->triangles[i])
			//	{
			//		if (mesh->triangles[i]->v1 == )
			//	}
			//}

			maxCount = newMaxCount;

			// truncate vertexExist array down
			vertexExist = (bool*)realloc(vertexExist, maxCount * sizeof(bool));

			// truncate rawVertices array down
			rawVertices = (Vertex*)realloc(rawVertices, maxCount * sizeof(Vertex));

			// truncate trsVertices array down
			trsVertices = (Vertex*)realloc(trsVertices, maxCount * sizeof(Vertex));

			// update vertices current count
			currCount = 0u;
			for (unsigned int i = 0u; i < maxCount; ++i) if (vertexExist[i]) ++currCount;
		}
	}*/
	// << // -----------------------------------------|


	// >> [STRUCT] Mesh::TexcrdStorage ---------------|
	// -- // -- Mesh::TexcrdStorage::constructor -- //
	Mesh::TexcrdStorage::TexcrdStorage(Mesh* mesh, unsigned int maxTexcdsCount)
		:mesh(mesh),
		count(0u),
		capacity(maxTexcdsCount),
		Count(count),
		Capacity(capacity)
	{
		texcdsExist = (bool*)malloc(capacity * sizeof(Texcrd));
		for (unsigned int i = 0u; i < capacity; ++i) texcdsExist[i] = false;
		texcrds = (Texcrd*)malloc(capacity * sizeof(Texcrd));
	}
	Mesh::TexcrdStorage::~TexcrdStorage()
	{
		for (unsigned int i = 0u; i < capacity; ++i)
		{
			if (texcdsExist[i])
			{
				texcrds[i].~Texcrd();
			}
		}
		free(texcdsExist);
		free(texcrds);
	}
	// -- // -- Mesh::TexcrdStorage::operator -- //
	Texcrd* Mesh::TexcrdStorage::operator[](unsigned int index) const
	{
		return (texcdsExist[index]) ? &texcrds[index] : nullptr;
	}
	// -- // -- Mesh::TexcrdStorage::methods -- //
	Texcrd* Mesh::TexcrdStorage::CreateTexcd(float u, float v)
	{
		if (count >= capacity)
			return nullptr;

		for (unsigned int i = 0u; i < capacity; ++i)
		{
			if (!texcdsExist[i])
			{
				new (&texcrds[i]) Texcrd(u, v);
				texcdsExist[i] = true;
				++count;
				return &texcrds[i];
			}
		}
		return nullptr;
	}
	void Mesh::TexcrdStorage::DestroyAllTexcds(unsigned int newCapacity)
	{
		// delete all texcds
		for (unsigned int i = 0u; i < capacity; ++i)
		{
			if (texcdsExist[i])
			{
				texcrds[i].~Texcrd();
			}
			texcdsExist[i] = false;
		}

		if (newCapacity == capacity)
			return;

		// free up texcds' memory
		free(texcdsExist);
		free(texcrds);

		// allocate new sized memory
		capacity = newCapacity;
		count = 0u;
		texcrds = (Texcrd*)malloc(capacity * sizeof(Texcrd));
		texcdsExist = (bool*)malloc(capacity * sizeof(bool));
		for (unsigned int i = 0u; i < capacity; ++i) texcdsExist[i] = false;
	}
	// << // -----------------------------------------|


	// >> [STRUCT] Mesh::TriangleStorage -------------|
	// -- // -- TriangleStorage::constructor -- //
	Mesh::TriangleStorage::TriangleStorage(Mesh* mesh, unsigned int maxTrianglesCount)
		:mesh(mesh),
		count(0u),
		capacity(maxTrianglesCount),
		Count(count),
		Capacity(capacity)
	{
		triangleExist = (bool*)malloc(capacity * sizeof(bool));
		for (unsigned int i = 0u; i < capacity; ++i) triangleExist[i] = false;
		rawTriangles = (Triangle*)malloc(capacity * sizeof(Triangle));
		trsTriangles = (Triangle*)malloc(capacity * sizeof(Triangle));
	}
	Mesh::TriangleStorage::~TriangleStorage()
	{
		for (unsigned int i = 0u; i < capacity; ++i)
		{
			if (triangleExist[i])
			{
				rawTriangles[i].~Triangle();
				trsTriangles[i].~Triangle();
			}
		}
		free(triangleExist);
		free(rawTriangles);
		free(trsTriangles);
	}
	// -- // -- TriangleStorage::operator -- //
	Triangle* Mesh::TriangleStorage::operator[](unsigned int index)
	{
		return (triangleExist[index]) ? &rawTriangles[index] : nullptr;
	}
	const Triangle* Mesh::TriangleStorage::operator[](unsigned int index) const
	{
		return (triangleExist[index]) ? &rawTriangles[index] : nullptr;
	}
	// -- // -- TriangleStorage::methods -- //
	Triangle* Mesh::TriangleStorage::CreateTriangle(Vertex* const V1, Vertex* const V2, Vertex* const V3,
		Texcrd* const T1, Texcrd* const T2, Texcrd* const T3,
		Graphics::Color color)
	{
		if (!V1 || !V2 || !V3 || count >= capacity)
			return nullptr;

		for (unsigned int i = 0; i < capacity; ++i)
		{
			if (!triangleExist[i])
			{
				new (&rawTriangles[i]) Triangle(V1, V2, V3, T1, T2, T3, color);
				new (&trsTriangles[i]) Triangle
				(
					mesh->m_vertices.trsVertices + (V1 - mesh->m_vertices.rawVertices),
					mesh->m_vertices.trsVertices + (V2 - mesh->m_vertices.rawVertices),
					mesh->m_vertices.trsVertices + (V3 - mesh->m_vertices.rawVertices),
					T1, T2, T3,
					color
				);
				triangleExist[i] = true;
				++count;
				return &rawTriangles[i];
			}
		}
		return nullptr;
	}
	bool Mesh::TriangleStorage::DestroyTriangle(const Triangle* const triangle)
	{
		if (triangle == nullptr)
			return false;

		for (unsigned int i = 0u; i < capacity; ++i)
		{
			if (triangle == &rawTriangles[i])
			{
				rawTriangles[i].~Triangle();
				trsTriangles[i].~Triangle();
				triangleExist[i] = false;
				--count;
				return true;
			}
		}
		return false;
	}
	void Mesh::TriangleStorage::DestroyAllTriangles(unsigned int newCapacity)
	{
		// deconstruct all vertices 
		for (unsigned int i = 0u; i < capacity; ++i)
		{
			if (triangleExist[i])
			{
				rawTriangles[i].~Triangle();
				trsTriangles[i].~Triangle();
			}
			triangleExist[i] = false;
		}

		if (newCapacity == capacity)
			return;

		// free up triangles' memory space
		free(rawTriangles);
		free(trsTriangles);
		free(triangleExist);

		// allocate new sized memory
		capacity = newCapacity;
		count = 0u;
		rawTriangles = (Triangle*)malloc(capacity * sizeof(Triangle));
		trsTriangles = (Triangle*)malloc(capacity * sizeof(Triangle));
		triangleExist = (bool*)malloc(capacity * sizeof(bool));
		for (unsigned int i = 0u; i < capacity; ++i) triangleExist[i] = false;
	}

	/*void Mesh::TriangleStorage::SetMaxCount(unsigned int newMaxCount)
	{
		if (newMaxCount == maxCount)
			return;

		if (newMaxCount > maxCount)
		{
			maxCount = newMaxCount;

			// size vertexExist array up
			triangleExist = (bool*)realloc(triangleExist, maxCount * sizeof(bool));

			// size rawVertices array up
			rawTriangles = (Triangle*)realloc(rawTriangles, maxCount * sizeof(Triangle));

			// size trsVertices array up
			trsTriangles = (Triangle*)realloc(trsTriangles, maxCount * sizeof(Triangle));
		}
		else
		{
			maxCount = newMaxCount;

			// truncate vertexExist array down
			triangleExist = (bool*)realloc(triangleExist, maxCount * sizeof(bool));

			// truncate rawVertices array down
			rawTriangles = (Triangle*)realloc(rawTriangles, maxCount * sizeof(Triangle));

			// truncate trsVertices array down
			trsTriangles = (Triangle*)realloc(rawTriangles, maxCount * sizeof(Triangle));

			// update vertices current count
			currCount = 0u;
			for (unsigned int i = 0u; i < maxCount; ++i) if (triangleExist[i]) ++currCount;
		}
	}*/
	// << // -----------------------------------------|


	// -- Mesh::constructors -- //
	Mesh::Mesh(
		const size_t& id,
		Updatable* updatable,
		const ConStruct<Mesh>& conStruct)
		: RenderObject(id, updatable, conStruct)
		, Vertices(m_vertices)
		, Triangles(m_triangles)
		, Texcrds(m_texcrds)
		, m_vertices(this, conStruct.maxVerticesCount)
		, m_texcrds(this, conStruct.maxTexcrdsCount)
		, m_triangles(this, conStruct.maxTrianglesCount)
	{}
	Mesh::~Mesh()
	{
		UnloadTexture();
	}


	// -- Mesh::methods -- //
	// public:
	void Mesh::DestroyAllComponents()
	{
		m_triangles.DestroyAllTriangles();

		m_vertices.DestroyAllVertices();
		m_texcrds.DestroyAllTexcds();
	}
	void Mesh::DestroyAllComponents(unsigned int newVerticesCapacity, unsigned int newTrianglesCapacity)
	{
		m_triangles.DestroyAllTriangles(newTrianglesCapacity);
		m_vertices.DestroyAllVertices(newVerticesCapacity);
		m_texcrds.DestroyAllTexcds(newVerticesCapacity);
	}
	void Mesh::TransposeComponents()
	{
		// update bounding volume
		m_bounding_volume.Reset();

		Math::vec3<float> P[6];
		Math::vec3<float> vN[6];

		vN[0] = Math::vec3<float>(0.0f, 1.0f, 0.0f);	// top
		vN[1] = Math::vec3<float>(0.0f, -1.0f, 0.0f);	// bottom
		vN[2] = Math::vec3<float>(-1.0f, 0.0f, 0.0f);	// left
		vN[3] = Math::vec3<float>(1.0f, 0.0f, 0.0f);	// right
		vN[4] = Math::vec3<float>(0.0f, 0.0f, -1.0f);	// front
		vN[5] = Math::vec3<float>(0.0f, 0.0f, 1.0f);	// back

		for (int i = 0; i < 6; i++)
		{
			vN[i].RotateZYX(-m_rotation);
		}


		for (unsigned int i = 0u; i < m_vertices.Capacity; ++i)
		{
			if (!m_vertices[i])
				continue;

			m_vertices.trsVertices[i] = m_vertices.rawVertices[i];

			for (int j = 0; j < 6; j++)
			{
				Math::vec3<float> V = m_vertices.trsVertices[i];
				V += m_center;
				V *= m_scale;
				if (Math::vec3<float>::DotProduct(V - P[j], vN[j]) > 0.0f)
					P[j] = V;
			}
		}

		for (int i = 0; i < 6; i++)
		{
			P[i].RotateXYZ(m_rotation);
		}

		m_bounding_volume.min.x = P[2].x;
		m_bounding_volume.min.y = P[1].y;
		m_bounding_volume.min.z = P[4].z;
		m_bounding_volume.max.x = P[3].x;
		m_bounding_volume.max.y = P[0].y;
		m_bounding_volume.max.z = P[5].z;

		m_bounding_volume.min += m_position;
		m_bounding_volume.max += m_position;

		// update triangles
		for (unsigned int i = 0u; i < m_triangles.Capacity; ++i)
		{
			if (!m_triangles[i])
				continue;

			m_triangles.trsTriangles[i].normal = Math::vec3<float>::CrossProduct(
				*m_triangles.trsTriangles[i].v2 - *m_triangles.trsTriangles[i].v3,
				*m_triangles.trsTriangles[i].v2 - *m_triangles.trsTriangles[i].v1);
			m_triangles.trsTriangles[i].normal.Normalize();
			m_triangles.trsTriangles[i].color = m_triangles[i]->color;
		}
	}
	bool Mesh::LoadFromFile(std::string file)
	{
		return false;
		// THIS CODE DOESN'T WORK

		//// [>] Check if its .obj file
		//int dotPos = file.find('.');
		//std::string fileExt = file.substr(dotPos + 1, std::min((int)file.length() - dotPos + 1, 3));
		//if (fileExt != "obj")
		//	return false;
		//
		//std::ifstream ifs;
		//ifs.open(file);
		//if (ifs.fail())
		//	return false;

		//std::string dataType = "";

		//// [>] Count vertices number in file
		//unsigned int verticesCount = 0u;
		//unsigned int trianglesCount = 0u;
		//while (!ifs.eof())
		//{
		//	char line[256u];
		//	ifs.getline(line, 256u);
		//	ifs >> dataType;
		//	if (dataType == "v")
		//		++verticesCount;
		//	else if (dataType == "f")
		//		++trianglesCount;
		//}
		//vertices.DestroyAllVertices(verticesCount);
		//triangles.DestroyAllTriangles(trianglesCount);



		//if (ifs.is_open())
		//	ifs.close();

		//return true;

		/*while (!ifs.eof())
		{
			char line[128u];
			ifs.getline(line, 128u);

			if (line[0] == 'v')
			{
				ifs >> line[0] >> currV.x >> currV.y >> currV.z;
				vs.push_back(currV);
			}
			else if (line[0] == 'f')
			{
				int f[3];
				ifs >> line[0] >> f[0] >> f[1] >> f[2];

			}
		}*/

	}
	void Mesh::LoadTexture(const Texture& newTexture)
	{
		if (this->m_pTexture == nullptr)	this->m_pTexture = new Texture(newTexture);
		else							*this->m_pTexture = newTexture;
	}
	void Mesh::UnloadTexture()
	{
		if (m_pTexture)
		{
			delete m_pTexture;
			m_pTexture = nullptr;
		}
	}
	
	const Texture* Mesh::GetTexture() const
	{
		return m_pTexture;
	}

	void Mesh::Update()
	{
		TransposeComponents();
	}
	// --------------------------------------------|
}