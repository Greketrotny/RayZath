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
#include "cuda_plane.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		class CudaWorld
		{
		public:
			CudaObjectContainer<Texture, CudaTexture> textures;
			CudaObjectContainer<NormalMap, CudaNormalMap> normal_maps;
			CudaObjectContainer<MetalnessMap, CudaMetalnessMap> metalness_maps;
			CudaObjectContainer<SpecularityMap, CudaSpecularityMap> specularity_maps;
			CudaObjectContainer<RoughnessMap, CudaRoughnessMap> roughness_maps;
			CudaObjectContainer<EmissionMap, CudaEmissionMap> emission_maps;

			CudaObjectContainer<Material, CudaMaterial> materials;
			CudaObjectContainer<MeshStructure, CudaMeshStructure> mesh_structures;

			CudaObjectContainer<Camera, CudaCamera> cameras;

			CudaObjectContainer<PointLight, CudaPointLight> point_lights;
			CudaObjectContainer<SpotLight, CudaSpotLight> spot_lights;
			CudaObjectContainer<DirectLight, CudaDirectLight> direct_lights;

			CudaObjectContainerWithBVH<Mesh, CudaMesh> meshes;
			CudaObjectContainerWithBVH<Sphere, CudaSphere> spheres;
			CudaObjectContainer<Plane, CudaPlane> planes;

			CudaMaterial material;
			CudaMaterial* default_material;
		public:
			static HostPinnedMemory m_hpm;


		public:
			__host__ CudaWorld();
			__host__ CudaWorld(const CudaWorld&) = delete;
			__host__ CudaWorld(CudaWorld&&) = delete;
			__host__ ~CudaWorld();


		public:
			__host__ void ReconstructResources(
				World& hWorld,
				cudaStream_t& update_stream);
			__host__ void ReconstructObjects(
				World& hWorld,
				cudaStream_t& update_stream);
			__host__ void ReconstructCameras(
				World& hWorld,
				cudaStream_t& update_stream);
			__host__ void ReconstructAll(
				World& host_world,
				cudaStream_t& mirror_stream);
		private:
			__host__ void ReconstructMaterial(
				const CudaWorld& hCudaWorld,
				const Material& hMaterial,
				cudaStream_t& mirror_stream);
			__host__ void ReconstructDefaultMaterial(
				const CudaWorld& hCudaWorld,
				const Material& hMaterial,
				cudaStream_t& mirror_stream);


			// intersection methods
		public:
			__device__ bool ClosestLightIntersection(
				RayIntersection& intersection) const
			{
				bool hit = false;

				// [>] PointLights
				for (uint32_t i = 0u; i < point_lights.GetCount(); ++i)
				{
					const CudaPointLight* point_light = &point_lights[i];
					hit |= point_light->ClosestIntersection(intersection);
				}


				// [>] SpotLights
				for (uint32_t i = 0u; i < spot_lights.GetCount(); ++i)
				{
					const CudaSpotLight* spot_light = &spot_lights[i];
					hit |= spot_light->ClosestIntersection(intersection);
				}


				// [>] DirectLights
				if (!(intersection.ray.near_far.y < 3.402823466e+38f))
				{
					for (uint32_t i = 0u; i < direct_lights.GetCount(); ++i)
					{
						const CudaDirectLight* direct_light = &direct_lights[i];
						hit |= direct_light->ClosestIntersection(intersection);
					}
				}

				return hit;
			}
			__device__ bool ClosestObjectIntersection(
				RayIntersection& intersection) const
			{
				const CudaRenderObject* closest_object = nullptr;

				// ~~~~ linear search ~~~~
				//for (uint32_t i = 0u; i < spheres.GetContainer().GetCount(); ++i)
				//{
				//	const CudaSphere& sphere = spheres.GetContainer()[i];
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
					const CudaPlane* plane = &planes[i];
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
				intersection.behind_material = &material;

				// apply medium scattering
				bool hit = intersection.ray.material->ApplyScattering(
					intersection, rng);

				// try to find intersection with light
				hit &= !ClosestLightIntersection(intersection);

				// try to find closer intersection with scene object
				hit |= ClosestObjectIntersection(intersection);

				return hit;
			}

			__device__ ColorF AnyIntersection(
				const CudaRay& shadow_ray) const
			{
				ColorF shadow_mask(1.0f);

				// planes
				for (uint32_t i = 0u; i < planes.GetCount(); ++i)
				{
					const CudaPlane* plane = &planes[i];
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


			__device__ __inline__ CudaTexcrd CalculateTexcrd(const vec3f& direction) const
			{
				return CudaTexcrd(
					-(0.5f + (atan2f(direction.z, direction.x) / (2.0f * CUDART_PI_F))),
					0.5f + (asinf(direction.y) / CUDART_PI_F));
			}
		};
	}
}

#endif // !CUDA_WORLD_H