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

				// trace ray through scene
				TraceRay(thread, World, tracing_path, intersection, color_mask);
				//color_mask *= intersection.bvh_factor;

				// set value to depth and space buffers
				camera.CurrentDepthBuffer().SetValue(
					thread.in_grid,
					intersection.ray.length);
				camera.SpaceBuffer().SetValue(
					thread.in_grid,
					intersection.ray.origin + intersection.ray.direction * intersection.ray.length);

				if (!tracing_path.NextNodeAvailable())
					return;

				// generate next ray
				const float metalic_ratio =
					intersection.surface_material->GenerateNextRay(
						thread,
						intersection,
						ckernel->GetRNG());

				// multiply color mask by surface color according to material metalness
				color_mask.Blend(
					color_mask * intersection.fetched_color,
					metalic_ratio);

				do
				{
					// trace ray 
					TraceRay(thread, World, tracing_path, intersection, color_mask);
					//color_mask *= intersection.bvh_factor;

					if (!tracing_path.NextNodeAvailable())
						return;

					// generate next ray
					const float metalic_ratio =
						intersection.surface_material->GenerateNextRay(
							thread,
							intersection,
							ckernel->GetRNG());

					// multiply color mask by surface color according to material metalness
					color_mask.Blend(
						color_mask * intersection.fetched_color,
						metalic_ratio);

				} while (tracing_path.FindNextNodeToTrace());
			}
			__device__ void RenderCumulativePass(
				FullThread& thread,
				const CudaWorld& World,
				CudaCamera& camera,
				TracingPath& tracing_path,
				RayIntersection& intersection)
			{
				ColorF color_mask(1.0f);

				do
				{
					// trace ray through scene
					TraceRay(thread, World, tracing_path, intersection, color_mask);
					//color_mask *= intersection.bvh_factor;

					if (!tracing_path.NextNodeAvailable())
						return;

					// generate next ray
					const float metalic_ratio =
						intersection.surface_material->GenerateNextRay(
							thread,
							intersection,
							ckernel->GetRNG());

					// multiply color mask by surface color according to material metalness
					color_mask.Blend(
						color_mask * intersection.fetched_color,
						metalic_ratio);

				} while (tracing_path.FindNextNodeToTrace());
			}

			__device__ void TraceRay(
				FullThread& thread,
				const CudaWorld& world,
				TracingPath& tracing_path,
				RayIntersection& intersection,
				ColorF& color_mask)
			{
				// find closest intersection with the world
				const bool any_hit = world.ClosestIntersection(thread, intersection, ckernel->GetRNG());

				
				// [>] Add material emittance
				// fetch color and emission at given point on material surface
				intersection.fetched_color = 
					intersection.surface_material->GetColor(intersection.texcrd);
				intersection.fetched_emission =	
					intersection.surface_material->GetEmission(intersection.texcrd);

				if (intersection.fetched_emission > 0.0f)
				{	// intersection with emitting object

					tracing_path.finalColor +=
						color_mask *
						intersection.fetched_color *
						intersection.fetched_emission;
				}


				if (!any_hit)
				{	// nothing has been hit - terminate path

					tracing_path.EndPath();
					return;
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

				color_mask *=
					intersection.ray.material->GetColor() *
					cui_powf(intersection.ray.material->GetTransmission(), intersection.ray.length);



				// [>] Fetch metalic, specular and roughness from surface material
				// (needed for BRDF and next even estimation)
				intersection.fetched_metalic =
					intersection.surface_material->GetMetalic(intersection.texcrd);
				intersection.fetched_specular =
					intersection.surface_material->GetSpecular(intersection.texcrd);
				intersection.fetched_roughness = 
					intersection.surface_material->GetRoughness(intersection.texcrd);


				// calculate intersection point 
				// (needed for direct sampling and next ray generation)
				intersection.point =
					intersection.ray.origin +
					intersection.ray.direction *
					intersection.ray.length;


				// [>] Apply direct sampling
				if (intersection.surface_material->SampleDirect(thread, ckernel->GetRNG()))
				{
					// sample direct light
					const ColorF direct_light = DirectSampling(thread, world, intersection);

					// specular/metalic factor

					// s - specularity
					// m - metalness
					// c - surface color
					// 
					// L - directly sampled light
					// dL - diffuse light
					// msL - metalic specular light
					// nsL - nonmetalic specular light

					// t = dL + Blend(nsL, msL, m)
					// t = L*c*(1-s) + L*s + L*c*s*m - L*s*m
					// t = L*(c - c*s + s + c*s*m - s*m)
					// t = L*(c + s*(-c + 1 + c*m - m))
					// t = L*(c + s*(1-m)*(1-c)) = L * smf

					const ColorF smf =
						(intersection.fetched_color + 
						(ColorF(1.0f) - intersection.fetched_color) * 
							(1.0f - intersection.fetched_metalic) * 
							intersection.fetched_specular);

					// add direct light
					tracing_path.finalColor +=
						direct_light * smf * color_mask;
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

				ColorF total_light(0.0f);

				// [>] PointLights
				for (uint32_t index = 0u, tested = 0u;
					(index < world.point_lights.GetCapacity() && tested < world.point_lights.GetCount());
					++index)
				{
					const CudaPointLight* point_light = &world.point_lights[index];
					if (!point_light->Exist()) continue;
					++tested;

					// sample light
					const vec3f vPL = point_light->SampleDirection(
						intersection.point,
						thread,
						ckernel->GetRNG());
					const float dPL = vPL.Length();

					// sample brdf
					const float brdf = intersection.surface_material->BRDF(intersection, vPL / dPL);
					if (brdf < 1.0e-4f) continue;

					// distance factor (inverse square law)
					const float d_factor = 1.0f / ((dPL + 1.0f) * (dPL + 1.0f));

					// scatering factor
					const float sctr_factor = cui_expf(-dPL * intersection.ray.material->GetScattering());

					// calculate radiance at P
					const float radianceP = point_light->material.GetEmission() * d_factor * sctr_factor * brdf;
					if (radianceP < 1.0e-4f) continue;	// unimportant light contribution

					// cast shadow ray and calculate color contribution
					CudaRay shadowRay(intersection.point + intersection.surface_normal * 0.0001f, vPL, dPL);
					total_light += point_light->material.GetColor() * radianceP * world.AnyIntersection(shadowRay);
				}


				// [>] SpotLights
				for (uint32_t index = 0u, tested = 0u;
					(index < world.spot_lights.GetCapacity() && tested < world.spot_lights.GetCount());
					++index)
				{
					const CudaSpotLight* spot_light = &world.spot_lights[index];
					if (!spot_light->Exist()) continue;
					++tested;

					// sample light
					const vec3f vPL = spot_light->SampleDirection(
						intersection.point,
						thread,
						ckernel->GetRNG());
					const float dPL = vPL.Length();

					// sample brdf
					const float brdf = intersection.surface_material->BRDF(intersection, vPL / dPL);
					if (brdf < 1.0e-4f) continue;

					// distance factor (inverse square law)
					const float d_factor = 1.0f / ((dPL + 1.0f) * (dPL + 1.0f));

					// scattering factor
					const float sctr_factor = cui_expf(-dPL * intersection.ray.material->GetScattering());

					// beam illumination
					float beamIllum = 1.0f;
					const float LP_dot_D = vec3f::Similarity(-vPL, spot_light->direction);
					if (LP_dot_D < spot_light->cos_angle) beamIllum = 0.0f;
					else beamIllum = 1.0f;

					// calculate radiance at P
					const float radianceP = spot_light->material.GetEmission() * d_factor * sctr_factor * beamIllum * brdf;
					if (radianceP < 1.0e-4f) continue;	// unimportant light contribution

					// cast shadow ray and calculate color contribution
					const CudaRay shadowRay(intersection.point + intersection.surface_normal * 0.001f, vPL, dPL);
					total_light += spot_light->material.GetColor() * radianceP * world.AnyIntersection(shadowRay);
				}


				// [>] DirectLights
				for (uint32_t index = 0u, tested = 0u;
					(index < world.direct_lights.GetCapacity() && tested < world.direct_lights.GetCount());
					++index)
				{
					const CudaDirectLight* direct_light = &world.direct_lights[index];
					if (!direct_light->Exist()) continue;
					++tested;

					// sample light
					const vec3f vPL = direct_light->SampleDirection(
						intersection.point,
						thread,
						ckernel->GetRNG());

					// sample brdf
					const float brdf = intersection.surface_material->BRDF(intersection, vPL.Normalized());
					if (brdf < 1.0e-4f) continue;

					// calculate radiance at P
					const float radianceP = direct_light->material.GetEmission() * brdf;
					if (radianceP < 1.0e-4f) continue;	// unimportant light contribution

					// cast shadow ray and calculate color contribution
					CudaRay shadowRay(intersection.point + intersection.surface_normal * 0.0001f, vPL);
					total_light += direct_light->material.GetColor() * radianceP * world.AnyIntersection(shadowRay);
				}

				return total_light;
			}
		}
	}
}