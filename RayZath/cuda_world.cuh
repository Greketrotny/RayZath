#ifndef CUDA_WORLD_H
#define CUDA_WORLD_H

#include "world.h"

#include "cuda_object_container.cuh"
#include "cuda_bvh.cuh"

#include "cuda_camera.cuh"

#include "cuda_point_light.cuh"
#include "cuda_spot_light.cuh"
#include "cuda_direct_light.cuh"

#include "cuda_mesh.cuh"
#include "cuda_sphere.cuh"

namespace RayZath
{
	class CudaWorld
	{
	public:
		CudaObjectContainer<Camera, CudaCamera> cameras;
		CudaObjectContainer<PointLight, CudaPointLight> pointLights;
		CudaObjectContainer<SpotLight, CudaSpotLight> spotLights;
		CudaObjectContainer<DirectLight, CudaDirectLight> directLights;
		CudaObjectContainerWithBVH<Mesh, CudaMesh> meshes;
		CudaObjectContainerWithBVH<Sphere, CudaSphere> spheres;


	public:
		static HostPinnedMemory m_hpm;


	public:
		__host__ CudaWorld();
		__host__ CudaWorld(const CudaWorld&) = delete;
		__host__ CudaWorld(CudaWorld&&) = delete;
		__host__ ~CudaWorld();


	public:
		__host__ void Reconstruct(
			World& host_world,
			cudaStream_t& mirror_stream);
	};
}

#endif // !CUDA_WORLD_H