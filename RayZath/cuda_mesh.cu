#include "cuda_mesh.cuh"

#include "cuda_texture_types.h"
#include "texture_indirect_functions.h"

namespace RayZath
{
	// ~~~~~~~~ [STRUCT] CudaVertexStorage ~~~~~~~~
	// -- CudaVertexStorage::constructor -- //
	__host__ CudaVertexStorage::CudaVertexStorage()
		: verticesMemory(nullptr)
		, vertexExist(nullptr)
		, capacity(0)
		, count(0)
	{
	}
	__host__ CudaVertexStorage::~CudaVertexStorage()
	{
		if (this->verticesMemory)	CudaErrorCheck(cudaFree(this->verticesMemory));
		if (this->vertexExist)		CudaErrorCheck(cudaFree(this->vertexExist));

		this->verticesMemory = nullptr;
		this->vertexExist = nullptr;
		capacity = 0;
		count = 0;
	}

	__host__ void CudaVertexStorage::Reconstruct(
		const Mesh::VertexStorage& hostVertices,
		HostPinnedMemory& hostPinnedMemory,
		cudaStream_t* mirrorStream)
	{
		count = hostVertices.Count;

		if (hostVertices.Capacity != this->capacity)
		{//--> verticesCapacities don't match

			// free vertices arrays
			if (this->verticesMemory) CudaErrorCheck(cudaFree(this->verticesMemory));
			if (this->vertexExist)   CudaErrorCheck(cudaFree(this->vertexExist));

			// update vertices current count and capacity
			this->count = hostVertices.Count;
			this->capacity = hostVertices.Capacity;

			// allocate new amounts memory for vertices arrays
			CudaErrorCheck(cudaMalloc(&this->verticesMemory, this->capacity * sizeof(CudaVertex)));
			CudaErrorCheck(cudaMalloc(&this->vertexExist, this->capacity * sizeof(bool)));

			// copy vertices data from hostMesh to cudaMesh
			CudaVertex* hostCudaVertices = (CudaVertex*)malloc(this->capacity * sizeof(CudaVertex));
			bool* hostCudaVerticesExist = (bool*)malloc(this->capacity * sizeof(bool));
			for (unsigned int i = 0u; i < this->capacity; ++i)
			{
				if (hostVertices[i])
				{
					new (&hostCudaVertices[i]) CudaVertex(hostVertices.trsVertices[i]);
					hostCudaVerticesExist[i] = true;
				}
				else
				{
					hostCudaVerticesExist[i] = false;
				}
			}
			CudaErrorCheck(cudaMemcpy(this->verticesMemory, hostCudaVertices, this->capacity * sizeof(CudaVertex), cudaMemcpyKind::cudaMemcpyHostToDevice));
			CudaErrorCheck(cudaMemcpy(this->vertexExist, hostCudaVerticesExist, this->capacity * sizeof(bool), cudaMemcpyKind::cudaMemcpyHostToDevice));
			free(hostCudaVertices);
			free(hostCudaVerticesExist);
		}
		else
		{// VerticesCapacities match so perform asnynchronous copying

			// divide work into chunks of vertices to fit in hostPinned memory
			int chunkSize = hostPinnedMemory.GetSize() / (sizeof(*verticesMemory) + sizeof(*vertexExist));
			if (chunkSize == 0) return;	// TODO: throw exception (too few memory for async copying)

			// reconstruct each vertex
			for (int startIndex = 0, endIndex; startIndex < this->capacity; startIndex += chunkSize)
			{
				if (startIndex + chunkSize > this->capacity) chunkSize = this->capacity - startIndex;
				endIndex = startIndex + chunkSize;

				// copy vertices from device memory
				CudaVertex* const hostCudaVertices = (CudaVertex*)hostPinnedMemory.GetPointerToMemory();
				CudaErrorCheck(cudaMemcpyAsync(hostCudaVertices, this->verticesMemory + startIndex, chunkSize * sizeof(CudaVertex), cudaMemcpyKind::cudaMemcpyDeviceToHost, *mirrorStream));
				CudaErrorCheck(cudaStreamSynchronize(*mirrorStream));
				bool* const hostCudaVerticesExist = (bool*)(hostCudaVertices + chunkSize);

				// loop through all vertices in the chunk
				for (int i = startIndex, j = 0; i < endIndex; ++i, ++j)
				{
					if (hostVertices[i])
					{
						new (&hostCudaVertices[j]) CudaVertex(hostVertices.trsVertices[i]);
						hostCudaVerticesExist[j] = true;
					}
					else
					{
						hostCudaVerticesExist[j] = false;
					}

				}

				// copy mirrored vertices back to device
				CudaErrorCheck(cudaMemcpyAsync(this->verticesMemory + startIndex, hostCudaVertices, chunkSize * sizeof(*verticesMemory), cudaMemcpyKind::cudaMemcpyHostToDevice, *mirrorStream));
				CudaErrorCheck(cudaStreamSynchronize(*mirrorStream));
				CudaErrorCheck(cudaMemcpyAsync(this->vertexExist + startIndex, hostCudaVerticesExist, chunkSize * sizeof(*vertexExist), cudaMemcpyKind::cudaMemcpyHostToDevice, *mirrorStream));
				CudaErrorCheck(cudaStreamSynchronize(*mirrorStream));
			}
		}
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	// ~~~~~~~~ [STRUCT] CudaTexcrdStorage ~~~~~~~~
	// -- CudaTexcrdStorage::constructor -- //
	__host__ CudaTexcrdStorage::CudaTexcrdStorage()
		: texcrdsMemory(nullptr)
		, texcrdExist(nullptr)
		, capacity(0), count(0)
	{
	}
	__host__ CudaTexcrdStorage::~CudaTexcrdStorage()
	{
		if (this->texcrdsMemory)	CudaErrorCheck(cudaFree(this->texcrdsMemory));
		if (this->texcrdExist)		CudaErrorCheck(cudaFree(this->texcrdExist));

		this->texcrdsMemory = nullptr;
		this->texcrdExist = nullptr;
		capacity = 0;
		count = 0;
	}

	__host__ void CudaTexcrdStorage::Reconstruct(
		const Mesh::TexcrdStorage& hostTexcrds,
		HostPinnedMemory& hostPinnedMemory,
		cudaStream_t* mirrorStream)
	{
		count = hostTexcrds.Count;

		if (hostTexcrds.Capacity != this->capacity)
		{//--> texcrdsCapacities don't match

			// free texcrds arrays
			if (this->texcrdsMemory) CudaErrorCheck(cudaFree(this->texcrdsMemory));
			if (this->texcrdExist)   CudaErrorCheck(cudaFree(this->texcrdExist));

			// update vertices current count and capacity
			this->count = hostTexcrds.Count;
			this->capacity = hostTexcrds.Capacity;

			// allocate new amounts memory for vertices array
			CudaErrorCheck(cudaMalloc(&this->texcrdsMemory, this->capacity * sizeof(*this->texcrdsMemory)));
			CudaErrorCheck(cudaMalloc(&this->texcrdExist, this->capacity * sizeof(*this->texcrdExist)));

			// copy vertices data from hostMesh to cudaMesh
			CudaTexcrd* hostCudaTexcrds = (CudaTexcrd*)malloc(this->capacity * sizeof(*this->texcrdsMemory));
			bool* hostCudaTexcrdsExist = (bool*)malloc(this->capacity * sizeof(*this->texcrdExist));
			for (unsigned int i = 0u; i < this->capacity; ++i)
			{
				if (hostTexcrds[i])
				{
					new (&hostCudaTexcrds[i]) CudaTexcrd(hostTexcrds.texcrds[i]);
					hostCudaTexcrdsExist[i] = true;
				}
				else
				{
					hostCudaTexcrdsExist[i] = false;
				}
			}
			CudaErrorCheck(cudaMemcpy(this->texcrdsMemory, hostCudaTexcrds, this->capacity * sizeof(*this->texcrdsMemory), cudaMemcpyKind::cudaMemcpyHostToDevice));
			CudaErrorCheck(cudaMemcpy(this->texcrdExist, hostCudaTexcrdsExist, this->capacity * sizeof(*this->texcrdExist), cudaMemcpyKind::cudaMemcpyHostToDevice));
			free(hostCudaTexcrds);
			free(hostCudaTexcrdsExist);
		}
		else
		{// VerticesCapacities match so perform asnynchronous copying

			// divide work into chunks of vertices to fit in hostPinned memory
			int chunkSize = hostPinnedMemory.GetSize() / (sizeof(*this->texcrdsMemory) + sizeof(*this->texcrdExist));
			if (chunkSize == 0) return;	// TODO: throw exception (too few memory for async copying)

			// reconstruct each vertex
			for (int startIndex = 0, endIndex; startIndex < this->capacity; startIndex += chunkSize)
			{
				if (startIndex + chunkSize > this->capacity) chunkSize = this->capacity - startIndex;
				endIndex = startIndex + chunkSize;

				// copy vertices from device memory
				CudaTexcrd* const hostCudaTexcrds = (CudaTexcrd*)hostPinnedMemory.GetPointerToMemory();
				CudaErrorCheck(cudaMemcpyAsync(hostCudaTexcrds, this->texcrdsMemory + startIndex, chunkSize * sizeof(*this->texcrdsMemory), cudaMemcpyKind::cudaMemcpyDeviceToHost, *mirrorStream));
				CudaErrorCheck(cudaStreamSynchronize(*mirrorStream));
				bool* const hostCudaTexcrdsExist = (bool*)(hostCudaTexcrds + chunkSize);

				// loop through all vertices in the chunk
				for (int i = startIndex, j = 0; i < endIndex; ++i, ++j)
				{
					if (hostTexcrds[i])
					{
						new (&hostCudaTexcrds[j]) CudaTexcrd(hostTexcrds.texcrds[i]);
						hostCudaTexcrdsExist[j] = true;
					}
					else
					{
						hostCudaTexcrdsExist[j] = false;
					}

				}

				// copy mirrored vertices back to device
				CudaErrorCheck(cudaMemcpyAsync(this->texcrdsMemory + startIndex, hostCudaTexcrds, chunkSize * sizeof(*this->texcrdsMemory), cudaMemcpyKind::cudaMemcpyHostToDevice, *mirrorStream));
				CudaErrorCheck(cudaStreamSynchronize(*mirrorStream));
				CudaErrorCheck(cudaMemcpyAsync(this->texcrdExist + startIndex, hostCudaTexcrdsExist, chunkSize * sizeof(*this->texcrdExist), cudaMemcpyKind::cudaMemcpyHostToDevice, *mirrorStream));
				CudaErrorCheck(cudaStreamSynchronize(*mirrorStream));
			}
		}
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	// ~~~~~~~~ [STRUCT] CudaTriangleStorage ~~~~~~~~
	// -- CudaTriangleStorage::constructors -- //
	__host__ CudaTriangleStorage::CudaTriangleStorage()
		: trianglesMemory(nullptr)
		, triangleExist(nullptr)
		, capacity(0), count(0)
	{}
	__host__ CudaTriangleStorage::~CudaTriangleStorage()
	{
		if (this->trianglesMemory)	CudaErrorCheck(cudaFree(this->trianglesMemory));
		if (this->triangleExist)	CudaErrorCheck(cudaFree(this->triangleExist));

		this->trianglesMemory = nullptr;
		this->triangleExist = nullptr;
		capacity = 0;
		count = 0;
	}

	__host__ void CudaTriangleStorage::Reconstruct(
		const Mesh& hostMesh,
		CudaMesh& hostCudaMesh,
		HostPinnedMemory& hostPinnedMemory,
		cudaStream_t* mirrorStream)
	{
		count = hostMesh.Triangles.Count;

		if (hostMesh.Triangles.Capacity != this->capacity)
		{// trianglesCapacities don't match

			// free trianges arrays
			if (this->trianglesMemory)	CudaErrorCheck(cudaFree(this->trianglesMemory));
			if (this->triangleExist)	CudaErrorCheck(cudaFree(this->triangleExist));

			// update trianges current count and capacity
			this->count = hostMesh.Triangles.Count;
			this->capacity = hostMesh.Triangles.Capacity;

			// allocate new amounts memory for trianges arrays
			CudaErrorCheck(cudaMalloc(&this->trianglesMemory, this->capacity * sizeof(*this->trianglesMemory)));
			CudaErrorCheck(cudaMalloc(&this->triangleExist, this->capacity * sizeof(*this->triangleExist)));

			// copy triangles data from hostMesh to cudaMesh
			CudaTriangle* hostCudaTriangles = (CudaTriangle*)malloc(this->capacity * sizeof(*hostCudaTriangles));
			bool* hostCudaTrianglesExist = (bool*)malloc(this->capacity * sizeof(*hostCudaTrianglesExist));
			for (unsigned int i = 0u; i < this->capacity; ++i)
			{
				if (hostMesh.Triangles[i])
				{
					new (&hostCudaTriangles[i]) CudaTriangle(hostMesh.Triangles.trsTriangles[i]);

					if (hostMesh.Triangles.trsTriangles[i].v1) hostCudaTriangles[i].v1 =
						&hostCudaMesh.vertices[hostMesh.Triangles.GetTriangle<Mesh::TriangleStorage::TriangleTypeTransposed>(i)->v1 - hostMesh.Vertices.GetVertex<Mesh::VertexStorage::VertexTypeTransposed>(0)];
					if (hostMesh.Triangles.trsTriangles[i].v2) hostCudaTriangles[i].v2 =
						&hostCudaMesh.vertices[hostMesh.Triangles.GetTriangle<Mesh::TriangleStorage::TriangleTypeTransposed>(i)->v2 - hostMesh.Vertices.GetVertex<Mesh::VertexStorage::VertexTypeTransposed>(0)];
					if (hostMesh.Triangles.trsTriangles[i].v3) hostCudaTriangles[i].v3 =
						&hostCudaMesh.vertices[hostMesh.Triangles.GetTriangle<Mesh::TriangleStorage::TriangleTypeTransposed>(i)->v3 - hostMesh.Vertices.GetVertex<Mesh::VertexStorage::VertexTypeTransposed>(0)];

					if (hostMesh.Triangles.trsTriangles[i].t1) hostCudaTriangles[i].t1 =
						&hostCudaMesh.texcrds[hostMesh.Triangles.GetTriangle<Mesh::TriangleStorage::TriangleTypeTransposed>(i)->t1 - hostMesh.Texcrds[0]];
					if (hostMesh.Triangles.trsTriangles[i].t2) hostCudaTriangles[i].t2 =
						&hostCudaMesh.texcrds[hostMesh.Triangles.GetTriangle<Mesh::TriangleStorage::TriangleTypeTransposed>(i)->t2 - hostMesh.Texcrds[0]];
					if (hostMesh.Triangles.trsTriangles[i].t3) hostCudaTriangles[i].t3 =
						&hostCudaMesh.texcrds[hostMesh.Triangles.GetTriangle<Mesh::TriangleStorage::TriangleTypeTransposed>(i)->t3 - hostMesh.Texcrds[0]];

					hostCudaTrianglesExist[i] = true;
				}
				else
				{
					hostCudaTrianglesExist[i] = false;
				}
			}
			CudaErrorCheck(cudaMemcpy(this->trianglesMemory, hostCudaTriangles, this->capacity * sizeof(*this->trianglesMemory), cudaMemcpyKind::cudaMemcpyHostToDevice));
			CudaErrorCheck(cudaMemcpy(this->triangleExist, hostCudaTrianglesExist, this->capacity * sizeof(*this->triangleExist), cudaMemcpyKind::cudaMemcpyHostToDevice));
			free(hostCudaTriangles);
			free(hostCudaTrianglesExist);
		}
		else
		{// TrianglesCapacities match so perform asynchronous copying

			// divide work into chunks of vertices to fit in hostPinned memory
			int chunkSize = hostPinnedMemory.GetSize() / (sizeof(*trianglesMemory) + sizeof(*triangleExist));
			if (chunkSize == 0) return;	// TODO: throw exception (too few memory for async copying)

			// reconstruct each triangle
			for (int startIndex = 0, endIndex; startIndex < this->capacity; startIndex += chunkSize)
			{
				if (startIndex + chunkSize > this->capacity) chunkSize = this->capacity - startIndex;
				endIndex = startIndex + chunkSize;

				// copy triangles from device memory
				CudaTriangle* const hostCudaTriangles = (CudaTriangle*)hostPinnedMemory.GetPointerToMemory();
				CudaErrorCheck(cudaMemcpyAsync(hostCudaTriangles, this->trianglesMemory + startIndex, chunkSize * sizeof(CudaTriangle), cudaMemcpyKind::cudaMemcpyDeviceToHost, *mirrorStream));
				CudaErrorCheck(cudaStreamSynchronize(*mirrorStream));
				bool* const hostCudaTrianglesExist = (bool*)(hostCudaTriangles + chunkSize);

				// loop through all triangles in the chunk
				for (int i = startIndex, j = 0; i < endIndex; ++i, ++j)
				{
					if (hostMesh.Triangles[i])
					{
						new (&hostCudaTriangles[j]) CudaTriangle(hostMesh.Triangles.trsTriangles[i]);

						if (hostMesh.Triangles.trsTriangles[i].v1) hostCudaTriangles[j].v1 =
							&hostCudaMesh.vertices[hostMesh.Triangles.GetTriangle<Mesh::TriangleStorage::TriangleTypeTransposed>(i)->v1 - hostMesh.Vertices.GetVertex<Mesh::VertexStorage::VertexTypeTransposed>(0)];
						if (hostMesh.Triangles.trsTriangles[i].v2) hostCudaTriangles[j].v2 =
							&hostCudaMesh.vertices[hostMesh.Triangles.GetTriangle<Mesh::TriangleStorage::TriangleTypeTransposed>(i)->v2 - hostMesh.Vertices.GetVertex<Mesh::VertexStorage::VertexTypeTransposed>(0)];
						if (hostMesh.Triangles.trsTriangles[i].v3) hostCudaTriangles[j].v3 =
							&hostCudaMesh.vertices[hostMesh.Triangles.GetTriangle<Mesh::TriangleStorage::TriangleTypeTransposed>(i)->v3 - hostMesh.Vertices.GetVertex<Mesh::VertexStorage::VertexTypeTransposed>(0)];

						if (hostMesh.Triangles.trsTriangles[i].t1) hostCudaTriangles[j].t1 =
							&hostCudaMesh.texcrds[hostMesh.Triangles.GetTriangle<Mesh::TriangleStorage::TriangleTypeTransposed>(i)->t1 - hostMesh.Texcrds[0]];
						if (hostMesh.Triangles.trsTriangles[i].t2) hostCudaTriangles[j].t2 =
							&hostCudaMesh.texcrds[hostMesh.Triangles.GetTriangle<Mesh::TriangleStorage::TriangleTypeTransposed>(i)->t2 - hostMesh.Texcrds[0]];
						if (hostMesh.Triangles.trsTriangles[i].t3) hostCudaTriangles[j].t3 =
							&hostCudaMesh.texcrds[hostMesh.Triangles.GetTriangle<Mesh::TriangleStorage::TriangleTypeTransposed>(i)->t3 - hostMesh.Texcrds[0]];

						hostCudaTrianglesExist[j] = true;
					}
					else
					{
						hostCudaTrianglesExist[j] = false;
					}
				}

				// copy mirrored triangles bac to device
				CudaErrorCheck(cudaMemcpyAsync(this->trianglesMemory + startIndex, hostCudaTriangles, chunkSize * sizeof(*trianglesMemory), cudaMemcpyKind::cudaMemcpyHostToDevice, *mirrorStream));
				CudaErrorCheck(cudaStreamSynchronize(*mirrorStream));
				CudaErrorCheck(cudaMemcpyAsync(this->triangleExist + startIndex, hostCudaTrianglesExist, chunkSize * sizeof(*triangleExist), cudaMemcpyKind::cudaMemcpyHostToDevice, *mirrorStream));
				CudaErrorCheck(cudaStreamSynchronize(*mirrorStream));
			}
		}
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	// ~~~~~~~~ [CLASS] CudaMesh ~~~~~~~~
	HostPinnedMemory CudaMesh::hostPinnedMemory(0xFFFF);

	__host__ CudaMesh::CudaMesh()
		: vertices()
		, texcrds()
		, triangles()
		, texture(nullptr)
	{
	}
	__host__ CudaMesh::~CudaMesh()
	{
		// destroy all CudaMesh components
		DestroyTextures();
	}

	__host__ void CudaMesh::Reconstruct(
		Mesh& hMesh, 
		cudaStream_t& mirror_stream)
	{
		if (!hMesh.GetStateRegister().IsModified()) return;

		this->vertices.Reconstruct(hMesh.Vertices, hostPinnedMemory, &mirror_stream);
		this->texcrds.Reconstruct(hMesh.Texcrds, hostPinnedMemory, &mirror_stream);
		this->triangles.Reconstruct(hMesh, *this, hostPinnedMemory, &mirror_stream);

		this->position = hMesh.GetPosition();
		this->rotation = hMesh.GetRotation();
		this->center = hMesh.GetCenter();
		this->scale = hMesh.GetScale();
		this->material = hMesh.GetMaterial();
		this->boundingVolume = hMesh.GetBoundingBox();

		CudaMesh::MirrorTextures(hMesh, &mirror_stream);

		hMesh.GetStateRegister().MakeUnmodified();
	}

	__host__ void CudaMesh::MirrorTextures(const Mesh& hostMesh, cudaStream_t* mirrorStream)
	{
		if (hostMesh.GetTexture() != nullptr)
		{
			if (this->texture == nullptr)
			{
				// host created texture so device must too
				CudaTexture* hostCudaTexture = (CudaTexture*)malloc(sizeof(CudaTexture));
				new (hostCudaTexture) CudaTexture();

				hostCudaTexture->Reconstruct(*hostMesh.GetTexture(), *mirrorStream);

				CudaErrorCheck(cudaMalloc(&this->texture, sizeof(CudaTexture)));
				CudaErrorCheck(cudaMemcpy(
					this->texture, hostCudaTexture, 
					sizeof(*this->texture), 
					cudaMemcpyKind::cudaMemcpyHostToDevice));
				free(hostCudaTexture);
			}
			else
			{
				//if (!hostMesh.UpdateRequests.GetUpdateRequestState(Mesh::MeshUpdateRequestTexture))
				//	return;

				// on both sides is texture - only mirror
				CudaTexture* hostCudaTexture = (CudaTexture*)this->hostPinnedMemory.GetPointerToMemory();
				if (this->hostPinnedMemory.GetSize() < sizeof(CudaTexture)) return;	// TODO: throw an exception (to few host-pinned memory)

				CudaErrorCheck(cudaMemcpyAsync(
					hostCudaTexture, this->texture, 
					sizeof(CudaTexture), 
					cudaMemcpyKind::cudaMemcpyDeviceToHost, *mirrorStream));
				CudaErrorCheck(cudaStreamSynchronize(*mirrorStream));

				hostCudaTexture->Reconstruct(*hostMesh.GetTexture(), *mirrorStream);

				CudaErrorCheck(cudaMemcpy(
					this->texture, hostCudaTexture, 
					sizeof(CudaTexture), 
					cudaMemcpyKind::cudaMemcpyHostToDevice));
				CudaErrorCheck(cudaStreamSynchronize(*mirrorStream));
			}
		}
		else
		{
			if (this->texture != nullptr)
			{
				// host has unloaded texture so destroy texture on device
				CudaTexture* hostCudaTexture = (CudaTexture*)malloc(sizeof(CudaTexture));
				CudaErrorCheck(cudaMemcpy(
					hostCudaTexture, this->texture, 
					sizeof(CudaTexture), 
					cudaMemcpyKind::cudaMemcpyDeviceToHost));

				hostCudaTexture->~CudaTexture();

				CudaErrorCheck(cudaFree(this->texture));
				this->texture = nullptr;
				free(hostCudaTexture);
			}
		}
	}
	__host__ void CudaMesh::DestroyTextures()
	{
		if (this->texture)
		{
			CudaTexture* hostCudaTexture = (CudaTexture*)malloc(sizeof(CudaTexture));
			CudaErrorCheck(cudaMemcpy(
				hostCudaTexture, this->texture, 
				sizeof(CudaTexture), 
				cudaMemcpyKind::cudaMemcpyDeviceToHost));

			hostCudaTexture->~CudaTexture();

			free(hostCudaTexture);

			CudaErrorCheck(cudaFree(this->texture));
			this->texture = nullptr;
		}
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}