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
	// -- CudaMesh::fields -- //
	HostPinnedMemory CudaMesh::hostPinnedMemory(0xFFFF);

	// -- CudaMesh::constructor -- //
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

	// -- CudaMesh::methods -- //
	__host__ void CudaMesh::Reconstruct(
		const Mesh& hostMesh, 
		cudaStream_t& mirror_stream)
	{
		this->vertices.Reconstruct(hostMesh.Vertices, hostPinnedMemory, &mirror_stream);
		this->texcrds.Reconstruct(hostMesh.Texcrds, hostPinnedMemory, &mirror_stream);
		this->triangles.Reconstruct(hostMesh, *this, hostPinnedMemory, &mirror_stream);


		this->position = hostMesh.GetPosition();
		this->rotation = hostMesh.GetRotation();
		this->scale = hostMesh.GetScale();
		this->material = hostMesh.GetMaterial();
		this->boundingVolume = hostMesh.m_boundingVolume;

		// [>] Mirror CudaMesh components
		CudaMesh::MirrorTextures(hostMesh, &mirror_stream);
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

	// public:
	/*__device__ CudaColor<float> CudaMesh::TraceRefractionRay(CudaWorld *world, const CudaEngineKernel::CudaRay& ray, CudaRenderObject::IntersectionProperties& intersectProps, const int& depth)
	{
		cudaVec3<float> P, currP;
		CudaTriangle *triangle, *closestTriangle = nullptr;

		CudaRay objectSpaceRay = ray;
		objectSpaceRay.origin -= this->position;
		objectSpaceRay.origin.RotateZ(-rotation.z);
		objectSpaceRay.origin.RotateY(-rotation.y);
		objectSpaceRay.origin.RotateX(-rotation.x);
		objectSpaceRay.direction.RotateZ(-rotation.z);
		objectSpaceRay.direction.RotateY(-rotation.y);
		objectSpaceRay.direction.RotateX(-rotation.x);

		// [>] Calculate internal point of intersection with ray
		float length = objectSpaceRay.length;
		float currDistance = maxDistance;

		if (trianglesDoubleSided)
		{
			for (unsigned int ct = 0u, tc = 0u; (ct < trianglesCapacity && tc < trianglesCount); ++ct)
			{
				triangle = trianglesPtrs[ct];
				if (triangle == nullptr)
					continue;
				++tc;

				if (CudaMesh::RayTriangleIntersect<TriangleFacingDoubleSided>(objectSpaceRay, triangle, currP, currDistance, maxDistance))
				{
					if (currDistance < maxDistance)
					{
						maxDistance = currDistance;
						P = currP;
						closestTriangle = triangle;
					}
				}
			}
		}
		else
		{
			for (unsigned int ct = 0u, tc = 0u; (ct < trianglesCapacity && tc < trianglesCount); ++ct)
			{
				triangle = trianglesPtrs[ct];
				if (triangle == nullptr)
					continue;
				++tc;

				if (CudaMesh::RayTriangleIntersect<TriangleFacingApart>(objectSpaceRay, triangle, currP, currDistance, maxDistance))
				{
					if (currDistance < maxDistance)
					{
						maxDistance = currDistance;
						P = currP;
						closestTriangle = triangle;
					}
				}
			}
		}

		// debug
		#if defined(__CUDACC__)
		atomicAdd(&world->debugInfo.rayObjectIntersectTests, 1);
		#endif

		if (!closestTriangle)
		{
			// propably mesh is not valid (has holes or misintersection)
			return CudaColor<float>(1.0f, 1.0f, 1.0f);
		}


		// [>] Compute refraction ray
		float dirDotNormal = cudaVec3<float>::DotProduct(closestTriangle->normal, objectSpaceRay.direction);
		float indexesRatio = refractiveIndex;
		float k = 1.0f - indexesRatio * indexesRatio * (1.0f - dirDotNormal * dirDotNormal);


		if (objectSpaceRay.strength * materialFactor > world->renderSettings.materialFactorLowerThreshold && depth > 0)
		{
			if (k < 0.0f)
			{
				// debug
				#if defined(__CUDACC__)
				atomicAdd(&world->debugInfo.refractionRays, 1);
				#endif

				// total internal reflection
				cudaVec3<float> reversedNormal = cudaVec3<float>::Reverse(closestTriangle->normal);
				cudaVec3<float> nextInternalVector = reversedNormal * -2.0f * cudaVec3<float>::DotProduct(reversedNormal, objectSpaceRay.direction) + objectSpaceRay.direction;
				CudaRay nextInternalRay(P + reversedNormal * 0.001f, nextInternalVector, closestTriangle->color, objectSpaceRay.strength);
				return this->TraceRefractionRay(world, nextInternalRay, intersectProps, depth - 1);
			}
			else
			{
				// debug
				#if defined(__CUDACC__)
				atomicAdd(&world->debugInfo.refractionRays, 1);
				#endif


				//P.Rotate(rotation.x, rotation.y, rotation.z);
				//P += this->origin;
				//objProp.surfaceNormal.Rotate(rotation.x, rotation.y, rotation.z);


				// compute and trace ray escaping from object
				cudaVec3<float> reversedNormal = cudaVec3<float>::Reverse(closestTriangle->normal);
				cudaVec3<float> refractVector = objectSpaceRay.direction * indexesRatio + reversedNormal * (indexesRatio * dirDotNormal - sqrtf(k));

				CudaRay outRay(P - reversedNormal * 0.001f, refractVector, closestTriangle->color, objectSpaceRay.strength * materialFactor);
				outRay.origin.Rotate(rotation.x, rotation.y, rotation.z);
				outRay.origin += this->position;
				outRay.direction.Rotate(rotation.x, rotation.y, rotation.z);

				return CudaColor<float>::BlendProduct(CudaEngineKernel::TraceRay(world, outRay, intersectProps, depth - 1), FetchTexture(closestTriangle, P));
			}
		}
		else
		{
			// treat object as transparent
			return CudaColor<float>::BlendProduct(CudaEngineKernel::TraceRay(world, objectSpaceRay, intersectProps, depth - 1), FetchTexture(closestTriangle, P));
		}
	}*/
	/*__device__ CudaColor<float> CudaMesh::TraceTransparentRay(CudaWorld *world, const CudaRay& ray, CudaRenderObject::IntersectionProperties& intersectProps, const int& depth)
	{
		cudaVec3<float> P, currP;
		CudaTriangle *triangle, *closestTriangle = nullptr;

		// [>] Calculate internal point of intersection with ray
		float maxDistance = ray.length;
		float currDistance = maxDistance;

		if (this->trianglesDoubleSided)
		{
			// trace next world ray
			return CudaEngineKernel::TraceRay(world, ray, intersectProps, depth);
		}
		else
		{
			for (unsigned int ct = 0u, tc = 0u; (ct < trianglesCapacity && tc < trianglesCount); ++ct)
			{
				triangle = trianglesPtrs[ct];
				if (triangle == nullptr)
					continue;
				++tc;

				if (CudaMesh::RayTriangleIntersect<TriangleFacingApart>(ray, triangle, currP, currDistance, maxDistance))
				{
					if (currDistance < maxDistance)
					{
						maxDistance = currDistance;
						P = currP;
						closestTriangle = triangle;
					}
				}
			}

			if (closestTriangle)
			{
				if (ray.strength * materialFactor > world->renderSettings.materialFactorLowerThreshold && depth > 0)
				{
					CudaRay outRay(P + closestTriangle->normal * 0.001f, ray.direction, closestTriangle->color, ray.strength * materialFactor);
					return CudaColor<float>::BlendProduct(CudaEngineKernel::TraceRay(world, outRay, intersectProps, depth - 1), FetchTexture(closestTriangle, P));
				}
				else
				{
					return closestTriangle->color;
				}
			}
			else
			{
				return CudaEngineKernel::TraceRay(world, ray, intersectProps, depth - 1);
			}
		}
	}*/

	// private:
	template <> __device__ __inline__ bool CudaMesh::RayTriangleIntersectAndUV<CudaMesh::TriangleFacing::TriangleFacingToward>(
		const CudaRay& ray,
		CudaTriangle* triangle,
		cudaVec3<float>& P,
		float& currDistance,
		const float& maxDistance,
		float& u, float& v)
	{
		cudaVec3<float> v1v2 = *triangle->v2 - *triangle->v1;
		cudaVec3<float> v1v3 = *triangle->v3 - *triangle->v1;
		cudaVec3<float> pvec = cudaVec3<float>::CrossProduct(ray.direction, v1v3);

		float facing = cudaVec3<float>::DotProduct(v1v2, pvec);
		if (facing < 0.001f)
			return false;

		float invDet = 1.0f / facing;

		cudaVec3<float> tvec = ray.origin - *triangle->v1;
		u = cudaVec3<float>::DotProduct(tvec, pvec) * invDet;
		if (u < 0.0f || u > 1.0f)
			return false;

		cudaVec3<float> qvec = cudaVec3<float>::CrossProduct(tvec, v1v2);
		v = cudaVec3<float>::DotProduct(ray.direction, qvec) * invDet;
		if (v < 0.0f || u + v > 1.0f)
			return false;

		float t = cudaVec3<float>::DotProduct(v1v3, qvec) * invDet;

		P = ray.origin + ray.direction * t;
		currDistance = (P - ray.origin).Magnitude();
		//if (currDistance > maxDistance)
		//	return false;

		return true;
	}
	template <> __device__ __inline__ bool CudaMesh::RayTriangleIntersectAndUV<CudaMesh::TriangleFacing::TriangleFacingApart>(
		const CudaRay& ray,
		CudaTriangle* triangle,
		cudaVec3<float>& P,
		float& currDistance,
		const float& maxDistance,
		float& u, float& v)
	{
		cudaVec3<float> v1v2 = *triangle->v2 - *triangle->v1;
		cudaVec3<float> v1v3 = *triangle->v3 - *triangle->v1;
		cudaVec3<float> pvec = cudaVec3<float>::CrossProduct(ray.direction, v1v3);

		float facing = cudaVec3<float>::DotProduct(v1v2, pvec);
		if (facing > 0.001f)
			return false;

		float invDet = 1.0f / facing;

		cudaVec3<float> tvec = ray.origin - *triangle->v1;
		u = cudaVec3<float>::DotProduct(tvec, pvec) * invDet;
		if (u < 0.0f || u > 1.0f)
			return false;

		cudaVec3<float> qvec = cudaVec3<float>::CrossProduct(tvec, v1v2);
		v = cudaVec3<float>::DotProduct(ray.direction, qvec) * invDet;
		if (v < 0.0f || u + v > 1.0f)
			return false;

		float t = cudaVec3<float>::DotProduct(v1v3, qvec) * invDet;

		P = ray.origin + ray.direction * t;
		currDistance = (P - ray.origin).Magnitude();
		if (currDistance > maxDistance)
			return false;

		return true;
	}
	template <> __device__ __inline__ bool CudaMesh::RayTriangleIntersectAndUV<CudaMesh::TriangleFacing::TriangleFacingDoubleSided>(
		const CudaRay& ray,
		CudaTriangle* triangle,
		cudaVec3<float>& P,
		float& currDistance,
		const float& maxDistance,
		float& u, float& v)
	{
		cudaVec3<float> v1v2 = *triangle->v2 - *triangle->v1;
		cudaVec3<float> v1v3 = *triangle->v3 - *triangle->v1;
		cudaVec3<float> pvec = cudaVec3<float>::CrossProduct(ray.direction, v1v3);

		float facing = cudaVec3<float>::DotProduct(v1v2, pvec);
		if (facing < 0.001f && facing > -0.001f)
			return false;

		float invDet = 1.0f / facing;

		cudaVec3<float> tvec = ray.origin - *triangle->v1;
		u = cudaVec3<float>::DotProduct(tvec, pvec) * invDet;
		if (u < 0.0f || u > 1.0f)
			return false;

		cudaVec3<float> qvec = cudaVec3<float>::CrossProduct(tvec, v1v2);
		v = cudaVec3<float>::DotProduct(ray.direction, qvec) * invDet;
		if (v < 0.0f || u + v > 1.0f)
			return false;

		float t = cudaVec3<float>::DotProduct(v1v3, qvec) * invDet;

		P = ray.origin + ray.direction * t;
		currDistance = (P - ray.origin).Magnitude();
		if (currDistance > maxDistance)
			return false;

		return true;
	}


	__device__ __inline__ CudaColor<float> CudaMesh::FetchTextureWithUV(CudaTriangle* triangle, const float& a1, const float& a2)
	{
		if (this->texture == nullptr)
			return triangle->color;

		if (!triangle->t1 || !triangle->t2 || !triangle->t3)
			return triangle->color;

		float a3 = 1.0f - a1 - a2;
		float u = triangle->t1->u * a2 + triangle->t2->u * a1 + triangle->t3->u * a3;
		float v = triangle->t1->v * a2 + triangle->t2->v * a1 + triangle->t3->v * a3;

		float4 color;
		#if defined(__CUDACC__)	
		color = tex2D<float4>(this->texture->textureObject, u, v);
		#endif
		return CudaColor<float>(color.z, color.y, color.x);
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}