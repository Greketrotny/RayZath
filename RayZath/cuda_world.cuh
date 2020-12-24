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
	namespace CudaEngine
	{
		class CudaWorld
		{
		public:
			CudaObjectContainer<Texture, CudaTexture> textures;
			CudaObjectContainer<Material, CudaMaterial> materials;
			CudaObjectContainer<MeshStructure, CudaMeshStructure> mesh_structures;

			CudaObjectContainer<Camera, CudaCamera> cameras;

			CudaObjectContainer<PointLight, CudaPointLight> pointLights;
			CudaObjectContainer<SpotLight, CudaSpotLight> spotLights;
			CudaObjectContainer<DirectLight, CudaDirectLight> directLights;

			CudaObjectContainerWithBVH<Mesh, CudaMesh> meshes;
			CudaObjectContainerWithBVH<Sphere, CudaSphere> spheres;

			CudaMaterial material;
		public:
			static HostPinnedMemory m_hpm;


		public:
			__host__ CudaWorld() = default;
			__host__ CudaWorld(const CudaWorld&) = delete;
			__host__ CudaWorld(CudaWorld&&) = delete;
			__host__ ~CudaWorld() = default;


		public:
			__host__ void Reconstruct(
				World& host_world,
				cudaStream_t& mirror_stream);


			// intersection methods
			__device__ bool ClosestLightIntersection(
				RayIntersection& intersection) const
			{
				bool hit = false;

				// [>] PointLights
				for (uint32_t index = 0u, tested = 0u;
					(index < pointLights.GetCapacity() && tested < pointLights.GetCount());
					++index)
				{
					const CudaPointLight* pointLight = &pointLights[index];
					if (!pointLight->Exist()) continue;
					++tested;

					const cudaVec3<float> vPL = pointLight->position - intersection.ray.origin;
					const float dPL = vPL.Length();

					// check if light is close enough
					if (dPL >= intersection.ray.length) continue;
					// check if light is in front of ray
					if (cudaVec3<float>::DotProduct(vPL, intersection.ray.direction) < 0.0f) continue;


					const float dist = RayToPointDistance(intersection.ray, pointLight->position);
					if (dist < pointLight->size)
					{	// ray intersects with the light
						intersection.ray.length = dPL;
						intersection.material = &pointLight->material;
						hit = true;
					}
				}


				// [>] SpotLights
				for (uint32_t index = 0u, tested = 0u;
					(index < spotLights.GetCapacity() && tested < spotLights.GetCount());
					++index)
				{
					const CudaSpotLight* spotLight = &spotLights[index];
					if (!spotLight->Exist()) continue;
					++tested;

					const cudaVec3<float> vPL = spotLight->position - intersection.ray.origin;
					const float dPL = vPL.Length();

					if (dPL >= intersection.ray.length) continue;
					const float vPL_dot_vD = cudaVec3<float>::DotProduct(vPL, intersection.ray.direction);
					if (vPL_dot_vD < 0.0f) continue;

					const float dist = RayToPointDistance(intersection.ray, spotLight->position);
					if (dist < spotLight->size)
					{
						const float t_dist = sqrtf(
							(spotLight->size + spotLight->sharpness) *
							(spotLight->size + spotLight->sharpness) -
							dist * dist);

						const cudaVec3<float> test_point =
							intersection.ray.origin + intersection.ray.direction * vPL_dot_vD -
							intersection.ray.direction * t_dist;

						const float LP_dot_D = cudaVec3<float>::Similarity(
							test_point - spotLight->position, spotLight->direction);
						if (LP_dot_D > spotLight->cos_angle)
						{
							intersection.ray.length = dPL;
							intersection.material = &spotLight->material;
							hit = true;
						}
					}
				}


				// [>] DirectLights
				if (!(intersection.ray.length < 3.402823466e+38f))
				{
					for (uint32_t index = 0u, tested = 0u;
						(index < directLights.GetCapacity() && tested < directLights.GetCount());
						++index)
					{
						const CudaDirectLight* directLight = &directLights[index];
						if (!directLight->Exist()) continue;
						++tested;

						const float dot = cudaVec3<float>::DotProduct(
							intersection.ray.direction,
							-directLight->direction);
						if (dot > directLight->cos_angular_size)
						{
							intersection.material = &directLight->material;
							hit = true;
						}
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

					if (sphere->RayIntersect(currentIntersection))
					{
						closest_object = sphere;
					}
				}*/

				spheres.GetBVH().ClosestIntersection(
					intersection,
					closest_object);

				meshes.GetBVH().ClosestIntersection(
					intersection,
					closest_object);


				if (closest_object)
				{	
					// transpose intersection elements into word's space
					intersection.surface_normal /= closest_object->scale;
					intersection.surface_normal.RotateXYZ(closest_object->rotation);
					intersection.surface_normal.Normalize();

					intersection.mapped_normal /= closest_object->scale;
					intersection.mapped_normal.RotateXYZ(closest_object->rotation);
					intersection.mapped_normal.Normalize();

					intersection.point =
						intersection.ray.origin +
						intersection.ray.direction *
						intersection.ray.length;

					// find material behind intersection sufrace
					if (intersection.material == nullptr)
						intersection.material = &this->material;

					return true;
				}
				else
				{
					return false;
				}
			}
			__device__ bool ClosestIntersection(
				RayIntersection& intersection) const
			{
				bool l_hit = ClosestLightIntersection(intersection);
				bool o_hit = ClosestObjectIntersection(intersection);
				return l_hit || o_hit;
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

					total_shadow *= sphere->ShadowRayIntersect(shadow_ray);
					if (total_shadow < 0.0001f) return total_shadow;
				}*/

				total_shadow *= spheres.GetBVH().AnyIntersection(shadow_ray);
				if (total_shadow < 0.0001f) return total_shadow;

				total_shadow *= meshes.GetBVH().AnyIntersection(shadow_ray);
				if (total_shadow < 0.0001f) return total_shadow;

				return total_shadow;
			}
		};
	}
}

#endif // !CUDA_WORLD_H