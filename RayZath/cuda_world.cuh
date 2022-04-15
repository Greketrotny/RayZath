#ifndef CUDA_WORLD_H
#define CUDA_WORLD_H

#include "cuda_kernel_data.cuh"
#include "world.h"

#include "cuda_object_container.cuh"
#include "cuda_bvh.cuh"

#include "cuda_camera.cuh"

#include "cuda_spot_light.cuh"
#include "cuda_direct_light.cuh"

#include "cuda_mesh.cuh"

namespace RayZath::Cuda
{
	class World
	{
	public:
		ObjectContainer<RayZath::Engine::Texture, Texture> textures;
		ObjectContainer<RayZath::Engine::NormalMap, NormalMap> normal_maps;
		ObjectContainer<RayZath::Engine::MetalnessMap, MetalnessMap> metalness_maps;
		ObjectContainer<RayZath::Engine::RoughnessMap, RoughnessMap> roughness_maps;
		ObjectContainer<RayZath::Engine::EmissionMap, EmissionMap> emission_maps;

		ObjectContainer<RayZath::Engine::Material, Material> materials;
		ObjectContainer<RayZath::Engine::MeshStructure, MeshStructure> mesh_structures;

		ObjectContainer<RayZath::Engine::Camera, Camera> cameras;

		ObjectContainer<RayZath::Engine::SpotLight, SpotLight> spot_lights;
		ObjectContainer<RayZath::Engine::DirectLight, DirectLight> direct_lights;

		ObjectContainerWithBVH<RayZath::Engine::Mesh, Mesh> meshes;

		Material material;
		Material* default_material;
	public:
		static HostPinnedMemory m_hpm;


	public:
		__host__ World();
		__host__ World(const World&) = delete;
		__host__ World(World&&) = delete;
		__host__ ~World();


	public:
		__host__ void ReconstructResources(
			RayZath::Engine::World& hWorld,
			cudaStream_t& update_stream);
		__host__ void ReconstructObjects(
			RayZath::Engine::World& hWorld,
			cudaStream_t& update_stream);
		__host__ void ReconstructCameras(
			RayZath::Engine::World& hWorld,
			cudaStream_t& update_stream);
		__host__ void ReconstructAll(
			RayZath::Engine::World& hWorld,
			cudaStream_t& mirror_stream);
	private:
		__host__ void ReconstructMaterial(
			const RayZath::Engine::Material& hMaterial,
			cudaStream_t& mirror_stream);
		__host__ void ReconstructDefaultMaterial(
			const RayZath::Engine::Material& hMaterial,
			cudaStream_t& mirror_stream);


		// intersection methods
	public:
		__device__ bool ClosestObjectIntersection(
			RayIntersection& intersection) const
		{
			meshes.ClosestIntersection(intersection);

			if (intersection.closest_object) intersection.closest_object->analyzeIntersection(intersection);
			else intersection.texcrd = CalculateTexcrd(intersection.ray.direction);

			return intersection.closest_object != nullptr;
		}
		__device__ bool ClosestIntersection(RayIntersection& intersection, RNG& rng) const
		{
			// reset material to world material - farthest possible material
			intersection.surface_material = &material;
			intersection.behind_material = &material;

			bool hit = false;
			// apply medium scattering
			hit |= intersection.ray.material->ApplyScattering(intersection, rng);
			// try to find closer intersection with scene object
			hit |= ClosestObjectIntersection(intersection);

			return hit;
		}

		__device__ ColorF AnyIntersection(const RangedRay& shadow_ray) const
		{
			return ColorF(1.0f) * meshes.AnyIntersection(shadow_ray);
		}

		__device__ bool SampleDirect(const Kernel::ConstantKernel& ckernel) const
		{
			return
				(spot_lights.GetCount() != 0u && ckernel.GetRenderConfig().GetLightSampling().GetSpotLight() != 0u) ||
				(direct_lights.GetCount() != 0u && ckernel.GetRenderConfig().GetLightSampling().GetDirectLight() != 0u);
		}
		__device__ __inline__ Texcrd CalculateTexcrd(const vec3f& direction) const
		{
			return Texcrd(
				-(0.5f + (atan2f(direction.z, direction.x) / (2.0f * CUDART_PI_F))),
				0.5f + (asinf(direction.y) / CUDART_PI_F));
		}
	};
}

#endif // !CUDA_WORLD_H