#ifndef CUDA_MESH_CUH
#define CUDA_MESH_CUH

#include "mesh.h"
#include "cuda_render_object.cuh"
#include "cuda_render_parts.cuh"

namespace RayZath
{
	// CudaMesh mesh structure structs
	typedef cudaVec3<float> CudaVertex;
	struct CudaVertexStorage
	{
		// -- fields -- //
	public:
		CudaVertex* verticesMemory;
		bool* vertexExist;
		unsigned int capacity, count;

		// -- constructor -- //
	public:
		__host__ CudaVertexStorage(void);
		__host__ ~CudaVertexStorage();

		// -- methods -- //
	public:
		__host__ void Reconstruct(
			const Mesh::VertexStorage& hostVertices,
			HostPinnedMemory& hostPinnedMemory,
			cudaStream_t* mirrorStream);

		__host__ __device__ __inline__ CudaVertex& operator[](unsigned int index)
		{
			return verticesMemory[index];
		}
		__host__ __device__ __inline__ bool ExistAt(unsigned int index)
		{
			return vertexExist[index];
		}
		__host__ __device__ __inline__ const unsigned int& GetCapacity()
		{
			return capacity;
		}
		__host__ __device__ __inline__ const unsigned int& GetCount()
		{
			return count;
		}
	};
	struct CudaTexcrdStorage
	{
		// -- fields -- //
	private:
		CudaTexcrd* texcrdsMemory;
		bool* texcrdExist;
		unsigned int capacity, count;

		// -- constructor -- //
	public:
		__host__ CudaTexcrdStorage(void);
		__host__ ~CudaTexcrdStorage();

		// -- methods -- //
	public:
		__host__ void Reconstruct(
			const Mesh::TexcrdStorage& hostTexcrds,
			HostPinnedMemory& hostPinnedMemory,
			cudaStream_t* mirrorStream);

		__host__ __device__ __inline__ CudaTexcrd& operator[](unsigned int index)
		{
			return texcrdsMemory[index];
		}
		__host__ __device__ __inline__ bool ExistAt(unsigned int index)
		{
			return texcrdExist[index];
		}
		__host__ __device__ __inline__ const unsigned int& GetCapacity()
		{
			return capacity;
		}
		__host__ __device__ __inline__ const unsigned int& GetCount()
		{
			return count;
		}
	};
	//struct CudaNormalsStorage
	//{
	//	// -- fields -- //
	//private:
	//	typedef cudaVec3<float> CudaNormal;
	//	CudaNormal *normalsMemory = nullptr;
	//	bool *normalExist = nullptr;
	//	int capacity, count;
	//	// -- constructor -- //
	//public:
	//	__host__ CudaNormalsStorage(/*TODO: put here host equivalent*/);
	//	__host__ ~CudaNormalsStorage();
	//	// -- methods -- //
	//public:
	//	__device__ __inline__ CudaNormal& operator[](unsigned int index);
	//	__device__ __inline__ bool ExistAt(unsigned int index);
	//};
	struct CudaTriangleStorage
	{
		// -- fields -- //
	private:
		CudaTriangle* trianglesMemory;
		bool* triangleExist;
		unsigned int capacity, count;

		// -- constructor -- //
	public:
		__host__ CudaTriangleStorage(void);
		__host__ ~CudaTriangleStorage();

		// -- methods -- //
	public:
		__host__ void Reconstruct(
			const Mesh& hostMesh,
			CudaMesh& hostCudaMesh,
			HostPinnedMemory& hostPinnedMemory,
			cudaStream_t* mirrorStream);

		__host__ __device__ __inline__ CudaTriangle& operator[](unsigned int index)
		{
			return trianglesMemory[index];
		}
		__host__ __device__ __inline__ const CudaTriangle& operator[](unsigned int index) const
		{
			return trianglesMemory[index];
		}
		__host__ __device__ __inline__ bool ExistAt(unsigned int index) const
		{
			return triangleExist[index];
		}
		__host__ __device__ __inline__ const unsigned int& GetCapacity() const
		{
			return capacity;
		}
		__host__ __device__ __inline__ const unsigned int& GetCount() const
		{
			return count;
		}
	};


	class CudaMesh : public CudaRenderObject
	{
	public:
		CudaVertexStorage vertices;		// |
		CudaTexcrdStorage texcrds;		// | initialization order of CudaMesh
		//Normals normals;				// | structure parts matters
		CudaTriangleStorage triangles;	// |

		CudaTexture* texture;
	private:
		static HostPinnedMemory hostPinnedMemory;


	public:
		__host__ CudaMesh(void);
		__host__ ~CudaMesh();


	public:
		__host__ void Reconstruct(const Mesh& hostMesh, cudaStream_t& mirror_stream);
	private:
		__host__ void MirrorTextures(const Mesh& hostMesh, cudaStream_t* mirrorStream);
		__host__ void DestroyTextures();


		// device rendering functions
	public:
		__device__ __inline__ bool RayIntersect(RayIntersection& intersection) const
		{
			// [>] transpose objectSpaceRay
			CudaRay objectSpaceRay = intersection.ray;
			objectSpaceRay.origin -= this->position;
			objectSpaceRay.origin.RotateZYX(-rotation);
			objectSpaceRay.direction.RotateZYX(-rotation);
			objectSpaceRay.origin /= this->scale;
			objectSpaceRay.direction /= this->scale;
			float length_factor = objectSpaceRay.direction.Magnitude();
			objectSpaceRay.length *= length_factor;
			objectSpaceRay.direction.Normalize();

			// [>] check ray intersection with boundingVolume
			if (!boundingVolume.RayIntersection(objectSpaceRay))
				return false;


			const CudaTriangle* triangle = nullptr, *closestTriangle = nullptr;
			cudaVec3<float> currP, objectPoint;

			float currTriangleDistance = objectSpaceRay.length;
			float currDistance = currTriangleDistance;
			float b1, b2;

			for (unsigned int index = 0u, tested = 0u; (index < triangles.GetCapacity() && tested < triangles.GetCount()); ++index)
			{
				triangle = &triangles[index];
				if (!triangles.ExistAt(index)) continue;
				++tested;

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

			for (unsigned int ct = 0u, tc = 0u; (ct < triangles.GetCapacity() && tc < triangles.GetCount()); ++ct)
			{
				triangle = &triangles[ct];
				if (!triangles.ExistAt(ct)) continue;
				++tc;

				if (CudaMesh::RayTriangleIntersectWithUV(
					objectSpaceRay, 
					triangle, 
					currP, 
					currTriangleDistance, currDistance,
					b1, b2))
				{
					const CudaColor<float> col = FetchTextureWithUV(triangle, b1, b2);
					shadow *= (1.0f - col.alpha);
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