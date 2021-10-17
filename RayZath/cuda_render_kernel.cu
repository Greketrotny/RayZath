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

				// get camera and clamp working threads
				CudaCamera* const camera = &world->cameras[camera_idx];
				if (thread.in_grid.x >= camera->GetWidth() ||
					thread.in_grid.y >= camera->GetHeight()) return;

				// create RNG
				RNG rng(
					vec2f(
						thread.in_grid.x / float(camera->GetWidth()),
						thread.in_grid.y / float(camera->GetHeight())),
					ckernel->GetSeeds().GetSeed(thread.in_grid_idx));

				// create intersection object
				RayIntersection intersection;
				intersection.ray.material = &world->material;

				// generate camera ray
				camera->GenerateSimpleRay(
					intersection.ray,
					thread,
					rng);

				// trace ray from camera
				TracingPath* tracingPath =
					&camera->GetTracingPath(thread.in_grid);
				tracingPath->ResetPath();

				RenderFirstPass(thread, *world, *camera, *tracingPath, intersection, rng);

				camera->SampleImageBuffer().SetValue(
					thread.in_grid,
					tracingPath->CalculateFinalColor());
				camera->PassesBuffer().SetValue(thread.in_grid, 1u);
			}
			__global__ void LaunchCumulativePass(
				CudaGlobalKernel* const global_kernel,
				CudaWorld* world,
				const int camera_idx)
			{
				// get kernels
				CudaGlobalKernel* const kernel = global_kernel;
				ckernel = &const_kernel[kernel->GetRenderIdx()];

				FullThread thread;

				// get camera and clamp working threads
				CudaCamera* const camera = &world->cameras[camera_idx];
				if (thread.in_grid.x >= camera->GetWidth() ||
					thread.in_grid.y >= camera->GetHeight()) return;

				// create RNG
				RNG rng(
					vec2f(
						thread.in_grid.x / float(camera->GetWidth()),
						thread.in_grid.y / float(camera->GetHeight())),
					ckernel->GetSeeds().GetSeed(thread.in_grid_idx));

				// create intersection object
				RayIntersection intersection;
				intersection.ray.material = &world->material;

				// generate camera ray
				camera->GenerateRay(
					intersection.ray,
					thread,
					rng);

				// get tracing path
				TracingPath* tracingPath =
					&camera->GetTracingPath(thread.in_grid);
				tracingPath->ResetPath();

				// render cumulative pass
				RenderCumulativePass(*world, *camera, *tracingPath, intersection, rng);
				camera->SampleImageBuffer().AppendValue(
					thread.in_grid,
					tracingPath->CalculateFinalColor());
				camera->PassesBuffer().AppendValue(thread.in_grid, 1u);
			}

			__device__ void RenderFirstPass(
				FullThread& thread,
				const CudaWorld& World,
				CudaCamera& camera,
				TracingPath& tracing_path,
				RayIntersection& intersection,
				RNG& rng)
			{
				// trace ray through scene
				TraceRay(World, tracing_path, intersection, rng);
				//color_mask *= intersection.bvh_factor;

				// set value to depth and space buffers
				camera.CurrentDepthBuffer().SetValue(
					thread.in_grid,
					intersection.ray.near_far.y);
				camera.SpaceBuffer().SetValue(
					thread.in_grid,
					intersection.ray.origin + intersection.ray.direction * intersection.ray.near_far.y);

				if (!tracing_path.NextNodeAvailable())
					return;

				// generate next ray
				const float metalness_ratio =
					intersection.surface_material->GenerateNextRay(
						intersection,
						rng);

				// multiply color mask by surface color according to material metalness
				intersection.ray.color.Blend(
					intersection.ray.color * intersection.fetched_color, 
					metalness_ratio);

				while (tracing_path.FindNextNodeToTrace())
				{
					// trace ray 
					TraceRay(World, tracing_path, intersection, rng);
					//color_mask *= intersection.bvh_factor;

					if (!tracing_path.NextNodeAvailable())
						return;

					// generate next ray
					const float metalness_ratio =
						intersection.surface_material->GenerateNextRay(
							intersection,
							rng);

					// multiply color mask by surface color according to material metalness
					intersection.ray.color.Blend(
						intersection.ray.color * intersection.fetched_color,
						metalness_ratio);

				}
			}
			__device__ void RenderCumulativePass(
				const CudaWorld& World,
				CudaCamera& camera,
				TracingPath& tracing_path,
				RayIntersection& intersection,
				RNG& rng)
			{
				do
				{
					// trace ray through scene
					TraceRay(World, tracing_path, intersection, rng);
					//color_mask *= intersection.bvh_factor;

					if (!tracing_path.NextNodeAvailable())
						return;

					// generate next ray
					const float metalness_ratio =
						intersection.surface_material->GenerateNextRay(
							intersection,
							rng);

					// multiply color mask by surface color according to material metalness
					intersection.ray.color.Blend(
						intersection.ray.color * intersection.fetched_color,
						metalness_ratio);

				} while (tracing_path.FindNextNodeToTrace());
			}

			__device__ void TraceRay(
				const CudaWorld& world,
				TracingPath& tracing_path,
				RayIntersection& intersection,
				RNG& rng)
			{
				// find closest intersection with the world
				const bool any_hit = world.ClosestIntersection(intersection, rng);


				// [>] Add material emittance
				// fetch color and emission at given point on material surface
				intersection.fetched_color =
					intersection.surface_material->GetOpacityColor(intersection.texcrd);
				intersection.fetched_emission =
					intersection.surface_material->GetEmission(intersection.texcrd);

				if (intersection.fetched_emission > 0.0f)
				{	// intersection with emitting object

					tracing_path.finalColor +=
						intersection.ray.color *
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

				intersection.ray.color *=
					intersection.ray.material->GetOpacityColor() *
					cui_powf(intersection.ray.material->GetOpacityColor().alpha, intersection.ray.near_far.y);



				// [>] Fetch metalness, specularity and roughness from surface material
				// (needed for BRDF and next even estimation)
				intersection.fetched_metalness =
					intersection.surface_material->GetMetalness(intersection.texcrd);
				intersection.fetched_specularity =
					intersection.surface_material->GetSpecularity(intersection.texcrd);
				intersection.fetched_roughness =
					intersection.surface_material->GetRoughness(intersection.texcrd);


				// calculate intersection point 
				// (needed for direct sampling and next ray generation)
				intersection.point =
					intersection.ray.origin +
					intersection.ray.direction *
					intersection.ray.near_far.y;


				// [>] Apply direct sampling
				if (intersection.surface_material->SampleDirect(intersection))
				{
					// sample direct light
					const ColorF direct_light = DirectSampling(world, intersection, rng);

					// specularity/metalness factor

					// s - specularity
					// m - metalness
					// c - surface color
					// 
					// L - directly sampled light
					// dL - diffuse light
					// msL - metalness specularity light
					// nsL - nonmetallic specularity light

					// t = dL + Blend(nsL, msL, m)
					// t = L*c*(1-s) + L*s + L*c*s*m - L*s*m
					// t = L*(c - c*s + s + c*s*m - s*m)
					// t = L*(c + s*(-c + 1 + c*m - m))
					// t = L*(c + s*(1-m)*(1-c)) = L * smf

					const ColorF smf =
						(intersection.fetched_color +
							(ColorF(1.0f) - intersection.fetched_color) *
							(1.0f - intersection.fetched_metalness) *
							intersection.fetched_specularity);

					// add direct light
					tracing_path.finalColor +=
						direct_light * smf * intersection.ray.color;
				}
			}

			__device__ Color<float> DirectSampling(
				const CudaWorld& world,
				RayIntersection& intersection,
				RNG& rng)
			{
				// Legend:
				// L - position of current light
				// P - point of intersetion
				// vN - surface normal

				ColorF total_light(0.0f);

				// [>] PointLights
				for (uint32_t i = 0u; i < world.point_lights.GetCount(); ++i)
				{
					const CudaPointLight* point_light = &world.point_lights[i];

					// sample light
					const vec3f vPL = point_light->SampleDirection(
						intersection.point,
						rng);
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
					const CudaRay shadowRay(intersection.point + intersection.surface_normal * 0.0001f, vPL, vec2f(0.0f, dPL));
					const ColorF shadow_color = world.AnyIntersection(shadowRay);
					total_light += point_light->material.GetColor() * shadow_color * shadow_color.alpha * radianceP;
				}


				// [>] SpotLights
				for (uint32_t i = 0u; i < world.spot_lights.GetCount(); ++i)
				{
					const CudaSpotLight* spot_light = &world.spot_lights[i];

					// sample light
					const vec3f vPL = spot_light->SampleDirection(
						intersection.point,
						rng);
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
					const CudaRay shadowRay(intersection.point + intersection.surface_normal * 0.001f, vPL, vec2f(0.0f, dPL));
					const ColorF shadow_color = world.AnyIntersection(shadowRay);
					total_light += spot_light->material.GetColor() * shadow_color * shadow_color.alpha * radianceP;
				}


				// [>] DirectLights
				for (uint32_t i = 0u; i < world.direct_lights.GetCount(); ++i)
				{
					const CudaDirectLight* direct_light = &world.direct_lights[i];

					// sample light
					const vec3f vPL = direct_light->SampleDirection(
						intersection.point,
						rng);

					// sample brdf
					const float brdf = intersection.surface_material->BRDF(intersection, vPL.Normalized());
					if (brdf < 1.0e-4f) continue;

					// calculate radiance at P
					const float radianceP = direct_light->material.GetEmission() * brdf;
					if (radianceP < 1.0e-4f) continue;	// unimportant light contribution

					// cast shadow ray and calculate color contribution
					const CudaRay shadowRay(intersection.point + intersection.surface_normal * 0.0001f, vPL);
					const ColorF shadow_color = world.AnyIntersection(shadowRay);
					total_light += direct_light->material.GetColor() * shadow_color * shadow_color.alpha * radianceP;
				}

				return total_light;
			}
		}
	}
}