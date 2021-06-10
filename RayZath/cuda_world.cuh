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
			CudaObjectContainer<MetalicMap, CudaMetalicMap> metalic_maps;
			CudaObjectContainer<SpecularMap, CudaSpecularMap> specular_maps;
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
				for (uint32_t index = 0u, tested = 0u;
					(index < point_lights.GetCapacity() && tested < point_lights.GetCount());
					++index)
				{
					const CudaPointLight* point_light = &point_lights[index];
					if (!point_light->Exist()) continue;
					++tested;

					hit |= point_light->ClosestIntersection(intersection);
				}


				// [>] SpotLights
				for (uint32_t index = 0u, tested = 0u;
					(index < spot_lights.GetCapacity() && tested < spot_lights.GetCount());
					++index)
				{
					const CudaSpotLight* spot_light = &spot_lights[index];
					if (!spot_light->Exist()) continue;
					++tested;

					hit |= spot_light->ClosestIntersection(intersection);
				}


				// [>] DirectLights
				if (!(intersection.ray.length < 3.402823466e+38f))
				{
					for (uint32_t index = 0u, tested = 0u;
						(index < direct_lights.GetCapacity() && tested < direct_lights.GetCount());
						++index)
					{
						const CudaDirectLight* direct_light = &direct_lights[index];
						if (!direct_light->Exist()) continue;
						++tested;

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
				/*// [>] Check every single sphere
				for (uint32_t index = 0u, tested = 0u;
					(index < World.spheres.GetContainer().GetCapacity() &&
						tested < World.spheres.GetContainer().GetCount());
					++index)
				{
					if (!World.spheres.GetContainer()[index].Exist()) continue;
					const CudaSphere* sphere = &World.spheres.GetContainer()[index];
					++tested;

					if (sphere->ClosestIntersection(currentIntersection))
					{
						closest_object = sphere;
					}
				}*/

				// spheres
				spheres.GetBVH().ClosestIntersection(
					intersection,
					closest_object);

				// meshes
				meshes.GetBVH().ClosestIntersection(
					intersection,
					closest_object);

				// planes
				for (uint32_t index = 0u, tested = 0u;
					(index < planes.GetCapacity() &&
						tested < planes.GetCount());
					++index)
				{
					if (!planes[index].Exist()) continue;
					const CudaPlane* plane = &planes[index];
					++tested;

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
				FullThread& thread,
				RayIntersection& intersection,
				const RNG& rng) const
			{
				// reset material to world material - farthest possible material
				intersection.surface_material = &material;

				// apply medium scattering
				const bool scattered = intersection.ray.material->ApplyScattering(
					thread, intersection, rng);

				// try to find intersection with light
				//ClosestLightIntersection(intersection);

				// try to find closer intersection with scene object
				const bool o_hit = ClosestObjectIntersection(intersection);

				if (intersection.behind_material == nullptr)
					intersection.behind_material = &material;

				return o_hit || scattered;
			}

			__device__ float AnyIntersection(
				const CudaRay& shadow_ray) const
			{
				float total_shadow = 1.0f;

				/*// [>] Test intersection with every sphere
				for (uint32_t index = 0u, tested = 0u;
					(index < world.spheres.GetContainer().GetCapacity() &&
						tested < world.spheres.GetContainer().GetCount());
					++index)
				{
					if (!world.spheres.GetContainer()[index].Exist()) continue;
					const CudaSphere* sphere = &world.spheres.GetContainer()[index];
					++tested;

					total_shadow *= sphere->AnyIntersection(shadow_ray);
					if (total_shadow < 0.0001f) return total_shadow;
				}*/

				// planes
				for (uint32_t index = 0u, tested = 0u;
					(index < planes.GetCapacity() &&
						tested < planes.GetCount());
					++index)
				{
					if (!planes[index].Exist()) continue;
					const CudaPlane* plane = &planes[index];
					++tested;

					total_shadow *= (plane->AnyIntersection(shadow_ray));
					if (total_shadow < 0.0001f) return total_shadow;
				}

				// spheres
				total_shadow *= spheres.GetBVH().AnyIntersection(shadow_ray);
				if (total_shadow < 0.0001f) return total_shadow;

				// meshes
				total_shadow *= meshes.GetBVH().AnyIntersection(shadow_ray);
				if (total_shadow < 0.0001f) return total_shadow;

				return total_shadow;
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