#ifndef CUDA_WORLD_H
#define CUDA_WORLD_H

#include "world.h"

#include "cuda_object_container.cuh"
#include "cuda_bvh.cuh"

#include "cuda_camera.cuh"

#include "cuda_spot_light.cuh"
#include "cuda_direct_light.cuh"

#include "cuda_mesh.cuh"
#include "cuda_sphere.cuh"
#include "cuda_plane.cuh"

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
		ObjectContainerWithBVH<RayZath::Engine::Sphere, Sphere> spheres;
		ObjectContainer<RayZath::Engine::Plane, Plane> planes;

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
			const RenderObject* closest_object = nullptr;

			// ~~~~ linear search ~~~~
			//for (uint32_t i = 0u; i < spheres.GetContainer().GetCount(); ++i)
			//{
			//	const Sphere& sphere = spheres.GetContainer()[i];
			//	if (sphere.ClosestIntersection(intersection))
			//	{
			//		closest_object = &sphere;
			//	}
			//}

			// spheres
			spheres.ClosestIntersection(
				intersection,
				closest_object);

			// meshes
			meshes.ClosestIntersection(
				intersection,
				closest_object);

			// planes
			for (uint32_t i = 0u; i < planes.GetCount(); ++i)
			{
				const Plane* plane = &planes[i];
				if (plane->ClosestIntersection(intersection))
				{
					closest_object = plane;
				}
			}


			if (closest_object)
			{
				closest_object->transformation.TransformVectorL2G(intersection.surface_normal);
				intersection.surface_normal.Normalize();

				closest_object->transformation.TransformVectorL2G(intersection.mapped_normal);
				intersection.mapped_normal.Normalize();

				return true;
			}
			else
			{
				intersection.texcrd = CalculateTexcrd(intersection.ray.direction);

				return false;
			}
		}
		__device__ bool ClosestIntersection(
			RayIntersection& intersection,
			RNG& rng) const
		{
			// reset material to world material - farthest possible material
			intersection.surface_material = &material;

			// apply medium scattering
			bool hit = intersection.ray.material->ApplyScattering(
				intersection, rng);

			// try to find closer intersection with scene object
			hit |= ClosestObjectIntersection(intersection);

			if (intersection.behind_material == nullptr)
				intersection.behind_material = &material;

			return hit;
		}

		__device__ ColorF AnyIntersection(
			const RangedRay& shadow_ray) const
		{
			ColorF shadow_mask(1.0f);

			// planes
			for (uint32_t i = 0u; i < planes.GetCount(); ++i)
			{
				const Plane* plane = &planes[i];
				shadow_mask *= (plane->AnyIntersection(shadow_ray));
				if (shadow_mask.alpha < 0.0001f) return shadow_mask;
			}

			// spheres
			shadow_mask *= spheres.AnyIntersection(shadow_ray);
			if (shadow_mask.alpha < 0.0001f) return shadow_mask;

			// meshes
			shadow_mask *= meshes.AnyIntersection(shadow_ray);
			if (shadow_mask.alpha < 0.0001f) return shadow_mask;

			return shadow_mask;
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