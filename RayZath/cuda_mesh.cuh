#ifndef CUDA_MESH_CUH
#define CUDA_MESH_CUH

#include "mesh.h"
#include "cuda_render_object.cuh"
#include "cuda_render_parts.cuh"

namespace RayZath
{
	typedef cudaVec3<float> CudaVertex;
	template <class HostComponent, class CudaComponent>
	struct CudaMeshComponentStorage
	{
	public:
		CudaComponent* memory;
		uint32_t capacity, count;


	public:
		__host__ CudaMeshComponentStorage()
			: memory(nullptr)
			, capacity(0u)
			, count(0u)
		{}
		__host__ ~CudaMeshComponentStorage()
		{
			if (memory) CudaErrorCheck(cudaFree(memory));
			memory = nullptr;

			capacity = 0u;
			count = 0u;
		}


	public:
		__host__ void Reconstruct(
			const ComponentStorage<HostComponent>& hComponents,
			HostPinnedMemory& hpm,
			cudaStream_t& mirror_stream)
		{
			count = hComponents.GetCount();

			if (hComponents.GetCapacity() != capacity)
			{//--> capacities don't match

				// free memory
				if (this->memory) CudaErrorCheck(cudaFree(memory));

				// update count and capacity
				this->count = hComponents.GetCount();
				this->capacity = hComponents.GetCapacity();

				// allocate new memory
				CudaErrorCheck(cudaMalloc(&memory, capacity * sizeof(CudaComponent)));

				// copy data from hostMesh to cudaMesh
				CudaComponent* hCudaComponents = (CudaComponent*)malloc(count * sizeof(CudaComponent));
				for (uint32_t i = 0u; i < count; ++i)
				{
					new (&hCudaComponents[i]) CudaComponent(hComponents[i]);
				}
				CudaErrorCheck(cudaMemcpy(
					memory, hCudaComponents, 
					count * sizeof(CudaComponent), 
					cudaMemcpyKind::cudaMemcpyHostToDevice));
				free(hCudaComponents);
			}
			else
			{// capacities match so perform asnynchronous copying

				// divide work into chunks of components to fit in host pinned memory
				int chunkSize = hpm.GetSize() / sizeof(*memory);
				if (chunkSize == 0) return;	// TODO: throw exception (too few memory for async copying)

				// reconstruct each component
				for (int startIndex = 0, endIndex; startIndex < count; startIndex += chunkSize)
				{
					if (startIndex + chunkSize > count) chunkSize = count - startIndex;
					endIndex = startIndex + chunkSize;

					// copy from device memory
					CudaComponent* const hCudaComponents = (CudaComponent*)hpm.GetPointerToMemory();
					CudaErrorCheck(cudaMemcpyAsync(
						hCudaComponents, memory + startIndex, 
						chunkSize * sizeof(CudaComponent), 
						cudaMemcpyKind::cudaMemcpyDeviceToHost, mirror_stream));
					CudaErrorCheck(cudaStreamSynchronize(mirror_stream));

					// loop through all components in the chunk
					for (int i = startIndex; i < endIndex; ++i)
					{
						new (&hCudaComponents[i]) CudaComponent(hComponents[i]);
					}

					// copy mirrored components back to device
					CudaErrorCheck(cudaMemcpyAsync(
						memory + startIndex, hCudaComponents, 
						chunkSize * sizeof(*memory), 
						cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
					CudaErrorCheck(cudaStreamSynchronize(mirror_stream));
				}
			}
		}

		__host__ __device__ __inline__ CudaComponent& operator[](const uint32_t& index)
		{
			return memory[index];
		}
		__host__ __device__ __inline__ const CudaComponent& operator[](const uint32_t& index) const
		{
			return memory[index];
		}
		__host__ __device__ __inline__ const uint32_t& GetCapacity() const
		{
			return capacity;
		}
		__host__ __device__ __inline__ const uint32_t& GetCount() const
		{
			return count;
		}
	};
	template<> struct CudaMeshComponentStorage<Triangle, CudaTriangle>
	{
	public:
		CudaTriangle* memory;
		uint32_t capacity, count;


	public:
		__host__ CudaMeshComponentStorage()
			: memory(nullptr)
			, capacity(0u)
			, count(0u)
		{}
		__host__ ~CudaMeshComponentStorage()
		{
			if (memory) CudaErrorCheck(cudaFree(memory));
			memory = nullptr;

			capacity = 0u;
			count = 0u;
		}


	public:
		__host__ void Reconstruct(
			const ComponentStorage<Triangle>& hTriangles,
			const ComponentStorage<Vertex>& hVertices,
			const ComponentStorage<Texcrd>& hTexcrds,
			CudaMeshComponentStorage<Vertex, CudaVertex>& hCudaVertices,
			CudaMeshComponentStorage<Texcrd, CudaTexcrd>& hCudaTexcrds,
			HostPinnedMemory& hpm,
			cudaStream_t& mirror_stream)
		{
			count = hTriangles.GetCount();

			if (hTriangles.GetCapacity() != capacity)
			{//--> capacities don't match

				// free memory
				if (this->memory) CudaErrorCheck(cudaFree(memory));

				// update count and capacity
				this->count = hTriangles.GetCount();
				this->capacity = hTriangles.GetCapacity();

				// allocate new memory
				CudaErrorCheck(cudaMalloc(&memory, capacity * sizeof(CudaTriangle)));

				// copy data from hostMesh to cudaMesh
				CudaTriangle* hCudaTriangles = (CudaTriangle*)malloc(count * sizeof(CudaTriangle));
				for (uint32_t i = 0u; i < count; ++i)
				{
					new (&hCudaTriangles[i]) CudaTriangle(hTriangles[i]);

					/*if (hostMesh.Triangles.trsTriangles[i].v1) hostCudaTriangles[i].v1 =
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
						&hostCudaMesh.texcrds[hostMesh.Triangles.GetTriangle<Mesh::TriangleStorage::TriangleTypeTransposed>(i)->t3 - hostMesh.Texcrds[0]];*/

					hCudaTriangles[i].v1 = &hCudaVertices[hTriangles[i].v1 - &hVertices[0]];
					hCudaTriangles[i].v2 = &hCudaVertices[hTriangles[i].v2 - &hVertices[0]];
					hCudaTriangles[i].v3 = &hCudaVertices[hTriangles[i].v3 - &hVertices[0]];

					hCudaTriangles[i].t1 = &hCudaTexcrds[hTriangles[i].t1 - &hTexcrds[0]];
					hCudaTriangles[i].t2 = &hCudaTexcrds[hTriangles[i].t2 - &hTexcrds[0]];
					hCudaTriangles[i].t3 = &hCudaTexcrds[hTriangles[i].t3 - &hTexcrds[0]];
				}
				CudaErrorCheck(cudaMemcpy(
					memory, hCudaTriangles,
					count * sizeof(CudaTriangle),
					cudaMemcpyKind::cudaMemcpyHostToDevice));
				free(hCudaTriangles);
			}
			else
			{// capacities match so perform asnynchronous copying

				// divide work into chunks of components to fit in host pinned memory
				int chunkSize = hpm.GetSize() / sizeof(*memory);
				if (chunkSize == 0) return;	// TODO: throw exception (too few memory for async copying)

				// reconstruct each component
				for (int startIndex = 0, endIndex; startIndex < count; startIndex += chunkSize)
				{
					if (startIndex + chunkSize > count) chunkSize = count - startIndex;
					endIndex = startIndex + chunkSize;

					// copy from device memory
					CudaTriangle* const hCudaTriangles = (CudaTriangle*)hpm.GetPointerToMemory();
					CudaErrorCheck(cudaMemcpyAsync(
						hCudaTriangles, memory + startIndex,
						chunkSize * sizeof(CudaTriangle),
						cudaMemcpyKind::cudaMemcpyDeviceToHost, mirror_stream));
					CudaErrorCheck(cudaStreamSynchronize(mirror_stream));

					// loop through all components in the chunk
					for (int i = startIndex; i < endIndex; ++i)
					{
						new (&hCudaTriangles[i]) CudaTriangle(hTriangles[i]);

						hCudaTriangles[i].v1 = &hCudaVertices[hTriangles[i].v1 - &hVertices[0]];
						hCudaTriangles[i].v2 = &hCudaVertices[hTriangles[i].v2 - &hVertices[0]];
						hCudaTriangles[i].v3 = &hCudaVertices[hTriangles[i].v3 - &hVertices[0]];

						hCudaTriangles[i].t1 = &hCudaTexcrds[hTriangles[i].t1 - &hTexcrds[0]];
						hCudaTriangles[i].t2 = &hCudaTexcrds[hTriangles[i].t2 - &hTexcrds[0]];
						hCudaTriangles[i].t3 = &hCudaTexcrds[hTriangles[i].t3 - &hTexcrds[0]];
					}

					// copy mirrored components back to device
					CudaErrorCheck(cudaMemcpyAsync(
						memory + startIndex, hCudaTriangles,
						chunkSize * sizeof(*memory),
						cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
					CudaErrorCheck(cudaStreamSynchronize(mirror_stream));
				}
			}
		}

		__host__ __device__ __inline__ CudaTriangle& operator[](const uint32_t& index)
		{
			return memory[index];
		}
		__host__ __device__ __inline__ const CudaTriangle& operator[](const uint32_t& index) const
		{
			return memory[index];
		}
		__host__ __device__ __inline__ const uint32_t& GetCapacity() const
		{
			return capacity;
		}
		__host__ __device__ __inline__ const uint32_t& GetCount() const
		{
			return count;
		}
	};

	class CudaMesh : public CudaRenderObject
	{
	public:
		CudaMeshComponentStorage<Vertex, CudaVertex> vertices;		// |
		CudaMeshComponentStorage<Texcrd, CudaTexcrd> texcrds;		// | initialization order of CudaMesh
		//CudaMeshComponentStorage<Normals, CudaNormals>;			// | structure parts matters
		CudaMeshComponentStorage<Triangle, CudaTriangle> triangles;	// |

		CudaTexture* texture;
	private:
		static HostPinnedMemory hostPinnedMemory;


	public:
		__host__ CudaMesh(void);
		__host__ ~CudaMesh();


	public:
		__host__ void Reconstruct(Mesh& hMesh, cudaStream_t& mirror_stream);
	private:
		__host__ void MirrorTextures(const Mesh& hostMesh, cudaStream_t* mirrorStream);
		__host__ void DestroyTextures();


		// device rendering functions
	public:
		__device__ __inline__ bool RayIntersect(RayIntersection& intersection) const
		{
			// [>] check ray intersection with boundingVolume
			if (!boundingVolume.RayIntersection(intersection.ray))
				return false;

			// [>] transpose objectSpaceRay
			CudaRay objectSpaceRay = intersection.ray;
			objectSpaceRay.origin -= this->position;
			objectSpaceRay.origin.RotateZYX(-rotation);
			objectSpaceRay.direction.RotateZYX(-rotation);
			objectSpaceRay.origin /= this->scale;
			objectSpaceRay.direction /= this->scale;
			objectSpaceRay.origin -= this->center;
			float length_factor = objectSpaceRay.direction.Magnitude();
			objectSpaceRay.length *= length_factor;
			objectSpaceRay.direction.Normalize();

			intersection.bvh_factor *= 0.9f;

			const CudaTriangle* triangle = nullptr, *closestTriangle = nullptr;
			cudaVec3<float> currP, objectPoint;

			float currTriangleDistance = objectSpaceRay.length;
			float currDistance = currTriangleDistance;
			float b1, b2;

			for (uint32_t index = 0u; index < triangles.GetCount(); ++index)
			{
				triangle = &triangles[index];

				if (CudaMesh::RayTriangleIntersectWithUV(
					objectSpaceRay, 
					triangle, 
					currP, currTriangleDistance, currDistance, 
					b1, b2))
				{
					if (currTriangleDistance < currDistance)
					{
						currDistance = currTriangleDistance;
						objectPoint = currP;
						closestTriangle = triangle;
					}
				}
			}

			if (closestTriangle)
			{
				intersection.surface_color = FetchTextureWithUV(closestTriangle, b1, b2);
				intersection.ray.length = currDistance / length_factor;

				// reverse normal if looking at back side of triangle
				cudaVec3<float> objectNormal = closestTriangle->normal;
				int reverse = cudaVec3<float>::DotProduct(
						objectNormal, 
						objectSpaceRay.direction) < 0.0f;
				objectNormal *= static_cast<float>((reverse ^ (reverse - 1)));

				// calculate world space normal
				intersection.normal = objectNormal;
				intersection.normal /= this->scale;
				intersection.normal.RotateXYZ(this->rotation);
				intersection.normal.Normalize();

				// calculate world space point of intersection
				intersection.point = objectPoint;
				intersection.point += this->center;
				intersection.point *= this->scale;
				intersection.point.RotateXYZ(this->rotation);
				intersection.point += this->position;

				const float transmitance =
					(1.0f - intersection.surface_color.alpha) * this->material.transmitance;

				// set material
				if (!reverse && transmitance > 0.0f)
				{	// intersection from inside

					// TODO: determine the material behind current material
					// or outer nested material we are currently in.
					// Now assumed to always be air/scene material (default one).
					intersection.material = CudaMaterial();
				}
				else
				{	// intersection from outside

					intersection.material = this->material;
					intersection.material.transmitance = transmitance;
				}


				return true;
			}

			return false;
		}
		__device__ __inline__ float ShadowRayIntersect(const CudaRay& ray) const
		{
			// [>] transpose objectSpaceRay
			CudaRay objectSpaceRay = ray;
			objectSpaceRay.origin -= this->position;
			objectSpaceRay.origin.RotateZYX(-rotation);
			objectSpaceRay.direction.RotateZYX(-rotation);
			objectSpaceRay.origin /= this->scale;
			objectSpaceRay.direction /= this->scale;
			objectSpaceRay.length *= objectSpaceRay.direction.Magnitude();
			objectSpaceRay.direction.Normalize();

			// [>] check ray intersection with boundingVolume
			if (!boundingVolume.RayIntersection(objectSpaceRay))
				return 1.0f;

			const CudaTriangle* triangle;
			cudaVec3<float> currP;

			float currTriangleDistance = objectSpaceRay.length;
			float currDistance = currTriangleDistance;
			float b1, b2;
			float shadow = this->material.transmitance;

			for (uint32_t index = 0u; index < triangles.GetCount(); ++index)
			{
				triangle = &triangles[index];

				if (CudaMesh::RayTriangleIntersectWithUV(
					objectSpaceRay, 
					triangle, 
					currP, 
					currTriangleDistance, currDistance,
					b1, b2))
				{
					const CudaColor<float> color = FetchTextureWithUV(triangle, b1, b2);
					shadow *= (1.0f - color.alpha);
					if (shadow < 0.0001f) return shadow;
				}
			}

			return 1.0f;
		}
	private:
		__device__ __inline__ bool RayTriangleIntersect(
			const CudaRay& ray,
			const CudaTriangle* triangle,
			cudaVec3<float>& P,
			float& currDistance,
			const float& maxDistance) const
		{
			// check triangle normal - ray direction similarity
			float triangleFacing = cudaVec3<float>::DotProduct(triangle->normal, ray.direction);
			if (triangleFacing < 0.001f && triangleFacing > -0.001f)
				return false;

			// check triangle position
			float D = cudaVec3<float>::DotProduct(triangle->normal, *triangle->v1);
			float T = -(cudaVec3<float>::DotProduct(triangle->normal, ray.origin) - D) / triangleFacing;
			if (T <= 0.0f)
				return false;

			// find point of intersection with triangle's plane
			P = ray.origin + ray.direction * T;

			// check if found point is closer or further from ray.origin by required minPointDistance
			currDistance = (P - ray.origin).Magnitude();
			if (currDistance > maxDistance)
				return false;

			// check if ray intersect triangle
			cudaVec3<float> C;
			cudaVec3<float> edge;
			cudaVec3<float> vp;

			edge = *triangle->v2 - *triangle->v1;
			vp = P - *triangle->v1;
			C = cudaVec3<float>::CrossProduct(edge, vp);
			if (cudaVec3<float>::DotProduct(triangle->normal, C) < 0.0f) return false;

			edge = *triangle->v3 - *triangle->v2;
			vp = P - *triangle->v2;
			C = cudaVec3<float>::CrossProduct(edge, vp);
			if (cudaVec3<float>::DotProduct(triangle->normal, C) < 0.0f) return false;

			edge = *triangle->v1 - *triangle->v3;
			vp = P - *triangle->v3;
			C = cudaVec3<float>::CrossProduct(edge, vp);
			if (cudaVec3<float>::DotProduct(triangle->normal, C) < 0.0f) return false;

			return true;
		}

		__device__ bool RayTriangleIntersectWithUV(
			const CudaRay& ray,
			const CudaTriangle* triangle,
			cudaVec3<float>& P,
			float& currDistance,
			const float& maxDistance,
			float& b1, float& b2) const
		{
			const cudaVec3<float> edge1 = *triangle->v2 - *triangle->v1;
			const cudaVec3<float> edge2 = *triangle->v3 - *triangle->v1;

			const cudaVec3<float> pvec = cudaVec3<float>::CrossProduct(ray.direction, edge2);

			const float det = (cudaVec3<float>::DotProduct(edge1, pvec));
			if (det > -0.0001f && det < 0.0001f)
				return false;

			const float inv_det = 1.0f / det;

			const cudaVec3<float> tvec = ray.origin - *triangle->v1;
			const float u = cudaVec3<float>::DotProduct(tvec, pvec) * inv_det;
			if (u < 0.0f || u > 1.0f)
				return false;

			const cudaVec3<float> qvec = cudaVec3<float>::CrossProduct(tvec, edge1);

			const float v = cudaVec3<float>::DotProduct(ray.direction, qvec) * inv_det;
			if (v < 0.0f || u + v > 1.0f)
				return false;

			const float t = cudaVec3<float>::DotProduct(edge2, qvec) * inv_det;
			if (t <= 0.0f)
				return false;

			P = ray.origin + ray.direction * t;

			currDistance = (P - ray.origin).Magnitude();
			if (currDistance > maxDistance)
				return false;

			b1 = u;
			b2 = v;

			return true;
		}


		__device__ CudaColor<float> FetchTexture(
			const CudaTriangle* triangle,
			const cudaVec3<float>& P) const
		{
			if (this->texture == nullptr)
				return triangle->color;

			if (!triangle->t1 || !triangle->t2 || !triangle->t3)
				return triangle->color;

			float Pv1 = (*triangle->v1 - P).Magnitude();
			float Pv2 = (*triangle->v2 - P).Magnitude();
			float Pv3 = (*triangle->v3 - P).Magnitude();

			float v1v2 = (*triangle->v1 - *triangle->v2).Magnitude();
			float v1v3 = (*triangle->v1 - *triangle->v3).Magnitude();
			float v2v3 = (*triangle->v2 - *triangle->v3).Magnitude();

			float Av1 = sqrtf((Pv3 + Pv2 + v2v3) * (-Pv3 + Pv2 + v2v3) * (Pv3 - Pv2 + v2v3) * (Pv3 + Pv2 - v2v3));
			float Av2 = sqrtf((Pv3 + Pv1 + v1v3) * (-Pv3 + Pv1 + v1v3) * (Pv3 - Pv1 + v1v3) * (Pv3 + Pv1 - v1v3));
			float Av3 = sqrtf((Pv1 + Pv2 + v1v2) * (-Pv1 + Pv2 + v1v2) * (Pv1 - Pv2 + v1v2) * (Pv1 + Pv2 - v1v2));
			float A = Av1 + Av2 + Av3;

			float u = (triangle->t1->u * Av1 + triangle->t2->u * Av2 + triangle->t3->u * Av3) / A;
			float v = (triangle->t1->v * Av1 + triangle->t2->v * Av2 + triangle->t3->v * Av3) / A;

			float4 color;
			#if defined(__CUDACC__)
			color = tex2D<float4>(this->texture->textureObject, u, v);
			#endif
			return CudaColor<float>(color.z, color.y, color.x);
		}
		__device__ CudaColor<float> FetchTextureWithUV(
			const CudaTriangle* triangle,
			const float& b1, const float& b2) const
		{
			if (this->texture == nullptr)
				return triangle->color;

			if (!triangle->t1 || !triangle->t2 || !triangle->t3)
				return triangle->color;

			const float b3 = 1.0f - b1 - b2;
			const float u = triangle->t1->u * b3 + triangle->t2->u * b1 + triangle->t3->u * b2;
			const float v = triangle->t1->v * b3 + triangle->t2->v * b1 + triangle->t3->v * b2;

			float4 color;
			#if defined(__CUDACC__)	
			color = tex2D<float4>(this->texture->textureObject, u, v);
			#endif
			return CudaColor<float>(color.z, color.y, color.x, color.w);
		}
	};
}

#endif // !CUDA_MESH_CUH