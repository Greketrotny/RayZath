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


			__global__ void GenerateCameraRay(
				CudaGlobalKernel* const global_kernel,
				CudaWorld* world,
				const int camera_id)
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
				ThreadData thread;
				thread.SetSeed(kernel->GetSeeds().GetSeed(thread.thread_in_block));

				// get camera and clamp working threads
				CudaCamera* const camera = &world->cameras[camera_id];
				if (thread.thread_x >= camera->GetWidth() || thread.thread_y >= camera->GetHeight()) return;


				// create intersection object
				RayIntersection intersection;
				intersection.ray.material = &world->material;

				// generate camera ray
				camera->GenerateRay(
					intersection.ray,
					thread,
					*ckernel);

				// trace ray from camera
				TracingPath* tracingPath =
					&camera->GetTracingPath(thread.thread_y * camera->GetWidth() + thread.thread_x);
				tracingPath->ResetPath();

				Render(thread, *world, *tracingPath, intersection);
				camera->AppendSample(tracingPath->CalculateFinalColor(), thread.thread_x, thread.thread_y);

				global_kernel->GetSeeds().SetSeed(thread.seed, thread.thread_in_block);
			}

			__device__ void Render(
				ThreadData& thread,
				const CudaWorld& World,
				TracingPath& tracing_path,
				RayIntersection& intersection)
			{
				Color<float> color_mask(1.0f);

				do
				{
					TraceRay(thread, World, tracing_path, intersection, color_mask);

					if (!tracing_path.NextNodeAvailable())
						return;

					intersection.surface_material->GenerateNextRay(
						thread,
						intersection,
						ckernel->GetRNG());

				} while (tracing_path.FindNextNodeToTrace());
			}

			__device__ void TraceRay(
				ThreadData& thread,
				const CudaWorld& world,
				TracingPath& tracing_path,
				RayIntersection& intersection,
				ColorF& color_mask)
			{
				intersection.surface_material = &world.material;

				if (intersection.ray.material->GetScattering() > 0.0f)
				{
					intersection.ray.length =
						(cui_logf(1.0f / (ckernel->GetRNG().GetUnsignedUniform(thread) + 1.0e-7f))) /
						intersection.ray.material->GetScattering();
				}


				if (!world.ClosestIntersection(intersection))
				{
					//color_mask *= intersection.bvh_factor;

					if (intersection.ray.material->GetScattering() > 0.0f)
					{
						// scattering point
						intersection.point =
							intersection.ray.origin +
							intersection.ray.direction * intersection.ray.length;

						// light illumination at scattering point
						tracing_path.finalColor +=
							color_mask *
							PointDirectSampling(thread, world, intersection);

						// generate scatter direction
						const vec3f sctr_direction = SampleSphere(
							ckernel->GetRNG().GetUnsignedUniform(thread),
							ckernel->GetRNG().GetUnsignedUniform(thread),
							intersection.ray.direction);

						// create scattering ray
						new (&intersection.ray) CudaSceneRay(
							intersection.point,
							sctr_direction,
							intersection.ray.material);
					}
					else
					{
						tracing_path.finalColor +=
							color_mask *
							intersection.surface_color *
							intersection.surface_material->GetEmittance();
						tracing_path.EndPath();
					}

					return;
				}

				//color_mask *= intersection.bvh_factor;


				// [>] Add material emittance
				if (intersection.surface_material->GetEmittance() > 0.0f)
				{	// intersection with emitting object

					tracing_path.finalColor +=
						color_mask *
						intersection.surface_color *
						intersection.surface_material->GetEmittance();
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
						SurfaceDirectSampling(thread, world, intersection);
				}
			}

			__device__ Color<float> SurfaceDirectSampling(
				ThreadData& thread,
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

					// dot product with surface normal
					const float vPL_dot_vN = vec3f::Similarity(vPL, intersection.mapped_normal);
					if (vPL_dot_vN <= 0.0f) continue;

					// distance factor (inverse square law)
					const float dPL = vPL.Length();
					const float d_factor = 1.0f / (dPL * dPL + 1.0f);

					// scatering factor
					const float sctr_factor = cui_expf(-dPL * intersection.ray.material->GetScattering());

					// calculate radiance at P
					const float radianceP = point_light->material.GetEmittance() * d_factor * sctr_factor * vPL_dot_vN;
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

					// dot product with surface normal
					const float vPL_dot_vN = vec3f::Similarity(vPL, intersection.mapped_normal);
					if (vPL_dot_vN <= 0.0f) continue;

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
					const float radianceP = spot_light->material.GetEmittance() * d_factor * sctr_factor * beamIllum * vPL_dot_vN;
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

					// dot product with sufrace normal
					const float vPL_dot_vN = vec3f::Similarity(vPL, intersection.mapped_normal);
					if (vPL_dot_vN <= 0.0f) continue;

					// calculate radiance at P
					const float radianceP = direct_light->material.GetEmittance() * vPL_dot_vN;
					if (radianceP < 0.0001f) continue;	// unimportant light contribution

					// cast shadow ray and calculate color contribution
					CudaRay shadowRay(intersection.point + intersection.surface_normal * 0.0001f, vPL);
					accLightColor += direct_light->material.GetColor() * radianceP * world.AnyIntersection(shadowRay);
				}

				return accLightColor;
			}
			__device__ Color<float> PointDirectSampling(
				ThreadData& thread,
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

					// distance factor (inverse square law)
					const float dPL = vPL.Length();
					const float d_factor = 1.0f / (dPL * dPL + 1.0f);

					// scattering factor
					const float sctr_factor = cui_expf(-dPL * intersection.ray.material->GetScattering());

					// calculate radiance at P
					const float radianceP = point_light->material.GetEmittance() * d_factor * sctr_factor;
					if (radianceP < 0.0001f) continue;	// unimportant light contribution

					// cast shadow ray and accumulate light contribution
					const CudaRay shadowRay(intersection.point, vPL, dPL);
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
					const float radianceP = spot_light->material.GetEmittance() * d_factor * sctr_factor * beamIllum;
					if (radianceP < 0.0001f) continue;	// unimportant light contribution

					// cast shadow ray and calculate color contribution
					const CudaRay shadowRay(intersection.point, vPL, dPL);
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

					// calculate light energy at P
					float energyAtP = direct_light->material.GetEmittance();
					if (energyAtP < 0.0001f) continue;	// unimportant light contribution

					// cast shadow ray and calculate color contribution
					CudaRay shadowRay(intersection.point, vPL);
					accLightColor += direct_light->material.GetColor() * energyAtP * world.AnyIntersection(shadowRay);
				}

				return accLightColor;
			}
		}
	}
}