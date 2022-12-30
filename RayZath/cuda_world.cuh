#ifndef CUDA_WORLD_H
#define CUDA_WORLD_H

#include "cuda_kernel_data.cuh"
#include "world.hpp"

#include "cuda_object_container.cuh"
#include "cuda_bvh.cuh"

#include "cuda_camera.cuh"

#include "cuda_spot_light.cuh"
#include "cuda_direct_light.cuh"

#include "cuda_instance.cuh"

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
		ObjectContainer<RayZath::Engine::Mesh, Mesh> meshes;

		ObjectContainer<RayZath::Engine::Camera, Camera> cameras;

		ObjectContainer<RayZath::Engine::SpotLight, SpotLight> spot_lights;
		ObjectContainer<RayZath::Engine::DirectLight, DirectLight> direct_lights;

		ObjectContainerWithBVH<RayZath::Engine::Instance, Instance> instances;

		Material material;
		Material* default_material;

		bool sample_direct = true;
		bool sample_direct_light = true;
		bool sample_spot_light = true;
		static HostPinnedMemory m_hpm;


	public:
		__host__ World();
		__host__ World(const World&) = delete;
		__host__ World(World&&) = delete;
		__host__ ~World();


	public:
		__host__ void reconstructResources(
			RayZath::Engine::World& hWorld,
			cudaStream_t& update_stream);
		__host__ void reconstructObjects(
			RayZath::Engine::World& hWorld,
			const RayZath::Engine::RenderConfig& render_config,
			cudaStream_t& update_stream);
		__host__ void reconstructCameras(
			RayZath::Engine::World& hWorld,
			cudaStream_t& update_stream);
		__host__ void reconstructAll(
			RayZath::Engine::World& hWorld,
			const RayZath::Engine::RenderConfig& render_config,
			cudaStream_t& mirror_stream);
	private:
		__host__ void reconstructMaterial(
			const RayZath::Engine::Material& hMaterial,
			cudaStream_t& mirror_stream);
		__host__ void reconstructDefaultMaterial(
			const RayZath::Engine::Material& hMaterial,
			cudaStream_t& mirror_stream);


		// intersection methods
	public:
		__device__ bool closestObjectIntersection(SceneRay& ray, SurfaceProperties& surface) const
		{
			TraversalResult traversal;
			instances.closestIntersection(ray, traversal);

			const bool found = traversal.closest_instance != nullptr;
			if (found) traversal.closest_instance->analyzeIntersection(traversal, surface);
			else surface.texcrd = calculateTexcrd(ray.direction);

			return found;
		}
		__device__ bool closestIntersection(SceneRay& ray, SurfaceProperties& surface, RNG& rng) const
		{
			bool hit = false;
			// apply medium scattering
			hit |= ray.material->applyScattering(ray, surface, rng);
			// try to find closer intersection with scene object
			hit |= closestObjectIntersection(ray, surface);

			return hit;
		}
		__device__ ColorF anyIntersection(const RangedRay& shadow_ray) const
		{
			return instances.anyIntersection(shadow_ray);
		}
		__device__ void rayCast(RangedRay& ray, uint32_t& object_idx, uint32_t& object_material_idx)
		{
			TraversalResult traversal;
			instances.closestIntersection(ray, traversal);
			if (traversal.closest_instance)
			{
				object_idx = traversal.closest_instance->m_instance_idx;
				if (traversal.closest_triangle)
					object_material_idx = traversal.closest_triangle->materialId();
			}
		}

		__device__ bool sampleDirect() const
		{
			return sample_direct;
		}
		__device__ __inline__ Texcrd calculateTexcrd(const vec3f& direction) const
		{
			return Texcrd(
				-(0.5f + (atan2f(direction.z, direction.x) / (2.0f * CUDART_PI_F))),
				0.5f + (asinf(direction.y) / CUDART_PI_F));
		}
	};
}

#endif // !CUDA_WORLD_H
