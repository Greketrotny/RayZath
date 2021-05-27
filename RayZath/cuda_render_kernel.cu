#include "cuda_render_kernel.cuh"

#include "math_constants.h"

namespace RayZath
{
	namespace CudaEngine
	{
		namespace CudaKernel
		{
			// ~~~~~~~~ Memory Management ~~~~~~~~
			__constant__ CudaConstantKernel const_kernel[2];
			__device__ const CudaConstantKernel* ckernel;

			__host__ void CopyToConstantMemory(
				const CudaConstantKernel* hCudaConstantKernel,
				const uint32_t& update_idx,
				cudaStream_t& stream)
			{
				CudaErrorCheck(cudaMemcpyToSymbolAsync(
					(const void*)const_kernel, hCudaConstantKernel,
					sizeof(CudaConstantKernel), update_idx * sizeof(CudaConstantKernel),
					cudaMemcpyKind::cudaMemcpyHostToDevice, stream));
				CudaErrorCheck(cudaStreamSynchronize(stream));
			}


			__global__ void LaunchFirstPass(
				CudaGlobalKernel* const global_kernel,
				CudaWorld* world,
				const int camera_idx)
			{
				/*// create local thread structure
				ThreadData thread(global_kernel->randomNumbers.GetSeed(threadIdx.y * blockDim.x + threadIdx.x));

				//CudaKernelData* const kernel = global_kernel;

				// [>] Copy kernel to shared memory
				extern __shared__ CudaKernelData shared_kernel[];
				CudaKernelData* kernel = shared_kernel;

				// copy render index
				if (thread.thread_in_kernel == 0u)
					kernel->renderIndex = global_kernel->renderIndex;

				// copy unsigned random floats
				const uint32_t linear_block_size = blockDim.x * blockDim.y;
				for (uint32_t i = thread.thread_in_block; i < RNG::s_count; i += linear_block_size)
				{
					kernel->randomNumbers.m_unsigned_uniform[i] =
						global_kernel->randomNumbers.m_unsigned_uniform[i];
				}

				__syncthreads();*/

				// get kernels
				CudaGlobalKernel* const kernel = global_kernel;
				ckernel = &const_kernel[kernel->GetRenderIdx()];

				// create thread object
				FullThread thread;
				thread.SetSeed(kernel->GetSeeds().GetSeed(thread.in_block_idx));

				// get camera and clamp working threads
				CudaCamera* const camera = &world->cameras[camera_idx];
				if (thread.in_grid.x >= camera->GetWidth() || 
					thread.in_grid.y >= camera->GetHeight()) return;

				// create intersection object
				RayIntersection intersection;
				intersection.ray.material = &world->material;

				// generate camera ray
				camera->GenerateSimpleRay(
					intersection.ray,
					thread,
					*ckernel);

				// trace ray from camera
				TracingPath* tracingPath =
					&camera->GetTracingPath(thread.in_grid);
				tracingPath->ResetPath();

				RenderFirstPass(thread, *world, *camera, *tracingPath, intersection);

				camera->SampleImageBuffer().SetValue(
					thread.in_grid,
					tracingPath->CalculateFinalColor());
				camera->PassesBuffer().SetValue(thread.in_grid, 1u);

				global_kernel->GetSeeds().SetSeed(thread.seed, thread.in_block_idx);
			}
			__global__ void LaunchCumulativePass(
				CudaGlobalKernel* const global_kernel,
				CudaWorld* world,
				const int camera_idx)
			{
				// get kernels
				CudaGlobalKernel* const kernel = global_kernel;
				ckernel = &const_kernel[kernel->GetRenderIdx()];

				// create thread object
				FullThread thread;
				thread.SetSeed(kernel->GetSeeds().GetSeed(thread.in_block_idx));

				// get camera and clamp working threads
				CudaCamera* const camera = &world->cameras[camera_idx];
				if (thread.in_grid.x >= camera->GetWidth() || 
					thread.in_grid.y >= camera->GetHeight()) return;

				// create intersection object
				RayIntersection intersection;
				intersection.ray.material = &world->material;

				// generate camera ray
				camera->GenerateRay(
					intersection.ray,
					thread,
					*ckernel);

				// get tracing path
				TracingPath* tracingPath =
					&camera->GetTracingPath(thread.in_grid);
				tracingPath->ResetPath();

				// render cumulative pass
				RenderCumulativePass(thread, *world, *camera, *tracingPath, intersection);
				camera->SampleImageBuffer().AppendValue(
					thread.in_grid,
					tracingPath->CalculateFinalColor());
				camera->PassesBuffer().AppendValue(thread.in_grid, 1u);

				global_kernel->GetSeeds().SetSeed(thread.seed, thread.in_block_idx);
			}

			__device__ void RenderFirstPass(
				FullThread& thread,
				const CudaWorld& World,
				CudaCamera& camera,
				TracingPath& tracing_path,
				RayIntersection& intersection)
			{
				Color<float> color_mask(1.0f);

				TraceRay(thread, World, tracing_path, intersection, color_mask);
				//color_mask *= intersection.bvh_factor;

				camera.CurrentDepthBuffer().SetValue(
					thread.in_grid,
					intersection.ray.length);
				camera.SpaceBuffer().SetValue(
					thread.in_grid,
					intersection.ray.origin + intersection.ray.direction * intersection.ray.length);

				if (!tracing_path.NextNodeAvailable())
					return;

				intersection.surface_material->GenerateNextRay(
					thread,
					intersection,
					ckernel->GetRNG());

				do
				{
					TraceRay(thread, World, tracing_path, intersection, color_mask);
					//color_mask *= intersection.bvh_factor;

					if (!tracing_path.NextNodeAvailable())
						return;

					intersection.surface_material->GenerateNextRay(
						thread,
						intersection,
						ckernel->GetRNG());

				} while (tracing_path.FindNextNodeToTrace());
			}
			__device__ void RenderCumulativePass(
				FullThread& thread,
				const CudaWorld& World,
				CudaCamera& camera,
				TracingPath& tracing_path,
				RayIntersection& intersection)
			{
				Color<float> color_mask(1.0f);

				do
				{
					TraceRay(thread, World, tracing_path, intersection, color_mask);
					//color_mask *= intersection.bvh_factor;

					if (!tracing_path.NextNodeAvailable())
						return;

					intersection.surface_material->GenerateNextRay(
						thread,
						intersection,
						ckernel->GetRNG());

				} while (tracing_path.FindNextNodeToTrace());
			}

			__device__ void TraceRay(
				FullThread& thread,
				const CudaWorld& world,
				TracingPath& tracing_path,
				RayIntersection& intersection,
				ColorF& color_mask)
			{
				// find closest intersection in the world
				if (!world.ClosestIntersection(thread, intersection, ckernel->GetRNG()))
				{
					tracing_path.finalColor +=
						color_mask *
						intersection.surface_color *
						intersection.surface_emittance;
					tracing_path.EndPath();
					return;
				}


				// calculate intersection point
				intersection.point =
					intersection.ray.origin +
					intersection.ray.direction *
					intersection.ray.length;


				// [>] Add material emittance
				if (intersection.surface_emittance > 0.0f)
				{	// intersection with emitting object

					tracing_path.finalColor +=
						color_mask *
						intersection.surface_color *
						intersection.surface_emittance;
				}


				// [>] Apply Beer's law

				// P0 - light energy in front of an object
				// P - light energy after going through an object
				// A - absorbance

				// e - material absorbance (constant)
				// b - distance traveled in an object
				// c - molar concentration (constant)

				// A = 10 ^ -(e * b * c)
				// P = P0 * A

				if (intersection.ray.material->GetTransmittance() > 0.0f)
				{
					color_mask *=
						intersection.surface_color *
						cui_powf(intersection.ray.material->GetTransmittance(), intersection.ray.length);
				}


				// [>] Apply direct sampling
				if (intersection.surface_material->SampleDirect(thread, ckernel->GetRNG()))
				{
					tracing_path.finalColor +=
						color_mask *
						intersection.surface_color *
						DirectSampling(thread, world, intersection);
				}
			}

			__device__ Color<float> DirectSampling(
				FullThread& thread,
				const CudaWorld& world,
				RayIntersection& intersection)
			{
				// Legend:
				// L - position of current light
				// P - point of intersetion
				// vN - surface normal

				Color<float> accLightColor(0.0f, 0.0f, 0.0f, 1.0f);

				// [>] PointLights
				for (uint32_t index = 0u, tested = 0u;
					(index < world.pointLights.GetCapacity() && tested < world.pointLights.GetCount());
					++index)
				{
					const CudaPointLight* point_light = &world.pointLights[index];
					if (!point_light->Exist()) continue;
					++tested;

					// sample light
					const vec3f vPL = point_light->SampleDirection(
						intersection.point,
						thread,
						ckernel->GetRNG());

					// sample brdf
					const float brdf = intersection.surface_material->BRDF(
						intersection.ray.direction, vPL, intersection.mapped_normal);
					if (brdf < 1.0e-4f) continue;

					// distance factor (inverse square law)
					const float dPL = vPL.Length();
					const float d_factor = 1.0f / (dPL * dPL + 1.0f);

					// scatering factor
					const float sctr_factor = cui_expf(-dPL * intersection.ray.material->GetScattering());

					// calculate radiance at P
					const float radianceP = point_light->material.GetEmittance() * d_factor * sctr_factor * brdf;
					if (radianceP < 0.0001f) continue;	// unimportant light contribution

					// cast shadow ray and calculate color contribution
					CudaRay shadowRay(intersection.point + intersection.surface_normal * 0.0001f, vPL, dPL);
					accLightColor += point_light->material.GetColor() * radianceP * world.AnyIntersection(shadowRay);
				}


				// [>] SpotLights
				for (uint32_t index = 0u, tested = 0u;
					(index < world.spotLights.GetCapacity() && tested < world.spotLights.GetCount());
					++index)
				{
					const CudaSpotLight* spot_light = &world.spotLights[index];
					if (!spot_light->Exist()) continue;
					++tested;

					// sample light
					const vec3f vPL = spot_light->SampleDirection(
						intersection.point,
						thread,
						ckernel->GetRNG());

					// sample brdf
					const float brdf = intersection.surface_material->BRDF(
						intersection.ray.direction, vPL, intersection.mapped_normal);
					if (brdf < 1.0e-4f) continue;

					// distance factor (inverse square law)
					const float dPL = vPL.Length();
					const float d_factor = 1.0f / (dPL * dPL + 1.0f);

					// scattering factor
					const float sctr_factor = cui_expf(-dPL * intersection.ray.material->GetScattering());

					// beam illumination
					float beamIllum = 1.0f;
					const float LP_dot_D = vec3f::Similarity(-vPL, spot_light->direction);
					if (LP_dot_D < spot_light->cos_angle) beamIllum = 0.0f;
					else beamIllum = 1.0f;

					// calculate radiance at P
					const float radianceP = spot_light->material.GetEmittance() * d_factor * sctr_factor * beamIllum * brdf;
					if (radianceP < 0.0001f) continue;	// unimportant light contribution

					// cast shadow ray and calculate color contribution
					const CudaRay shadowRay(intersection.point + intersection.surface_normal * 0.001f, vPL, dPL);
					accLightColor += spot_light->material.GetColor() * radianceP * world.AnyIntersection(shadowRay);
				}


				// [>] DirectLights
				for (uint32_t index = 0u, tested = 0u;
					(index < world.directLights.GetCapacity() && tested < world.directLights.GetCount());
					++index)
				{
					const CudaDirectLight* direct_light = &world.directLights[index];
					if (!direct_light->Exist()) continue;
					++tested;

					// sample light
					const vec3f vPL = direct_light->SampleDirection(
						intersection.point,
						thread,
						ckernel->GetRNG());

					// sample brdf
					const float brdf = intersection.surface_material->BRDF(
						intersection.ray.direction, vPL, intersection.mapped_normal);
					if (brdf < 1.0e-4f) continue;

					// calculate radiance at P
					const float radianceP = direct_light->material.GetEmittance() * brdf;
					if (radianceP < 0.0001f) continue;	// unimportant light contribution

					// cast shadow ray and calculate color contribution
					CudaRay shadowRay(intersection.point + intersection.surface_normal * 0.0001f, vPL);
					accLightColor += direct_light->material.GetColor() * radianceP * world.AnyIntersection(shadowRay);
				}

				return accLightColor;
			}
		}
	}
}