#include "cuda_render_kernel.cuh"

#include "math_constants.h"

namespace RayZath::Cuda::Kernel
{
	__global__ void RenderFirstPass(
		GlobalKernel* const global_kernel,
		World* const world,
		const uint32_t camera_idx)
	{
		FullThread thread;

		// get camera and clamp working threads
		Camera& camera = world->cameras[camera_idx];
		if (thread.in_grid.x >= camera.GetWidth() ||
			thread.in_grid.y >= camera.GetHeight()) return;

		// get kernels
		GlobalKernel& gkernel = *global_kernel;
		ConstantKernel& ckernel = const_kernel[gkernel.GetRenderIdx()];

		// create RNG
		RNG rng(
			vec2f(
				thread.in_grid.x / float(camera.GetWidth()),
				thread.in_grid.y / float(camera.GetHeight())),
			ckernel.GetSeeds().GetSeed(thread.in_grid_idx));

		// create intersection object
		RayIntersection intersection;
		intersection.ray = camera.GetTracingStates().GetRay(thread.in_grid);
		intersection.ray.near_far = camera.GetNearFar();

		// trace ray through scene
		TracingState tracing_state(ColorF(0.0f), 0u);
		const vec3f sample_direction = TraceRay(ckernel, *world, tracing_state, intersection, rng);

		// set depth
		camera.CurrentDepthBuffer().SetValue(
			thread.in_grid,
			intersection.ray.near_far.y);

		// set intersection point
		camera.SpaceBuffer().SetValue(
			thread.in_grid,
			intersection.ray.origin + intersection.ray.direction * intersection.ray.near_far.y);

		// set path depth
		camera.GetTracingStates().SetPathDepth(
			thread.in_grid,
			tracing_state.path_depth);

		// set color value
		tracing_state.final_color.alpha = 1.0f;
		camera.CurrentImageBuffer().SetValue(
			thread.in_grid,
			tracing_state.final_color);

		if (tracing_state.path_depth < ckernel.GetRenderConfig().GetTracing().GetMaxDepth())
		{
			intersection.ray.color.Blend(
				intersection.ray.color * intersection.color,
				intersection.next_ray_metalness);

			intersection.RepositionRay(sample_direction);
			camera.GetTracingStates().SetRay(thread.in_grid, intersection.ray);

			camera.AppendPathPos(thread.in_grid);
		}
	}
	__global__ void RenderRegeneratedPass(
		GlobalKernel* const global_kernel,
		World* const world,
		const uint32_t camera_idx)
	{
		FullThread thread;

		Camera& camera = world->cameras[camera_idx];
		if (thread.in_grid.x >= camera.GetWidth() ||
			thread.in_grid.y >= camera.GetHeight()) return;

		// get kernels
		GlobalKernel& gkernel = *global_kernel;
		ConstantKernel& ckernel = const_kernel[gkernel.GetRenderIdx()];

		TracingState tracing_state(
			ColorF(0.0f),
			camera.GetTracingStates().GetPathDepth(thread.in_grid));

		RNG rng(
			vec2f(
				thread.in_grid.x / float(camera.GetWidth()),
				thread.in_grid.y / float(camera.GetHeight())),
			ckernel.GetSeeds().GetSeed(thread.in_grid_idx + tracing_state.path_depth));

		// create intersection object
		RayIntersection intersection;
		intersection.ray = camera.GetTracingStates().GetRay(thread.in_grid);

		// trace ray through scene
		const vec3f sample_direction = TraceRay(ckernel, *world, tracing_state, intersection, rng);
		const bool path_continues = tracing_state.path_depth < ckernel.GetRenderConfig().GetTracing().GetMaxDepth();

		// update path depth
		camera.GetTracingStates().SetPathDepth(
			thread.in_grid,
			tracing_state.path_depth);

		// append additional light contribution passing along traced ray
		ColorF sample = camera.CurrentImageBuffer().GetValue(thread.in_grid);
		sample.red += tracing_state.final_color.red;
		sample.green += tracing_state.final_color.green;
		sample.blue += tracing_state.final_color.blue;
		sample.alpha += float(!path_continues);
		camera.CurrentImageBuffer().SetValue(thread.in_grid, sample);

		if (path_continues)
		{
			intersection.ray.color.Blend(
				intersection.ray.color * intersection.color,
				intersection.next_ray_metalness);

			intersection.RepositionRay(sample_direction);
			camera.GetTracingStates().SetRay(thread.in_grid, intersection.ray);

			camera.AppendPathPos(thread.in_grid);
		}
	}
	__global__ void RenderCumulativePass(
		GlobalKernel* const global_kernel,
		World* const world,
		const uint32_t camera_idx)
	{
		FullThread thread;

		Camera& camera = world->cameras[camera_idx];
		if (thread.in_grid.x >= camera.GetWidth() ||
			thread.in_grid.y >= camera.GetHeight()) return;
		if (thread.in_grid.y * camera.GetWidth() + thread.in_grid.x >= camera.GetTracingStates().GetPathCount())
			return;

		// get kernels
		GlobalKernel& gkernel = *global_kernel;
		ConstantKernel& ckernel = const_kernel[gkernel.GetRenderIdx()];

		const vec2ui32 path_pixel(vec2ui32(camera.GetTracingStates().GetPathPos(thread.in_grid)));
		TracingState tracing_state(
			ColorF(0.0f),
			camera.GetTracingStates().GetPathDepth(path_pixel));

		RNG rng(
			vec2f(
				thread.in_grid.x / float(camera.GetWidth()),
				thread.in_grid.y / float(camera.GetHeight())),
			ckernel.GetSeeds().GetSeed(thread.in_grid_idx + tracing_state.path_depth));

		// create intersection object
		RayIntersection intersection;
		intersection.ray = camera.GetTracingStates().GetRay(path_pixel);

		// trace ray through scene
		const vec3f sample_direction = TraceRay(ckernel, *world, tracing_state, intersection, rng);
		const bool path_continues = tracing_state.path_depth < ckernel.GetRenderConfig().GetTracing().GetMaxDepth();

		// update path depth
		camera.GetTracingStates().SetPathDepth(
			path_pixel,
			tracing_state.path_depth);

		// append additional light contribution passing along traced ray
		ColorF sample = camera.CurrentImageBuffer().GetValue(path_pixel);
		sample.red += tracing_state.final_color.red;
		sample.green += tracing_state.final_color.green;
		sample.blue += tracing_state.final_color.blue;
		sample.alpha += float(!path_continues);
		camera.CurrentImageBuffer().SetValue(path_pixel, sample);

		if (path_continues)
		{
			intersection.ray.color.Blend(
				intersection.ray.color * intersection.color,
				intersection.next_ray_metalness);

			intersection.RepositionRay(sample_direction);
			camera.GetTracingStates().SetRay(path_pixel, intersection.ray);

			camera.AppendPathPos(path_pixel);
		}
	}
	__global__ void RegenerateTerminatedRay(
		GlobalKernel* const global_kernel,
		World* const world,
		const uint32_t camera_idx)
	{
		Camera& camera = world->cameras[camera_idx];
		GridThread thread;
		if (thread.in_grid.x >= camera.GetWidth() ||
			thread.in_grid.y >= camera.GetHeight()) return;

		GlobalKernel& gkernel = *global_kernel;
		ConstantKernel& ckernel = const_kernel[gkernel.GetRenderIdx()];

		const uint8_t path_depth = camera.GetTracingStates().GetPathDepth(thread.in_grid);

		// create RNG
		RNG rng(
			vec2f(
				thread.in_grid.x / float(camera.GetWidth()),
				thread.in_grid.y / float(camera.GetHeight())),
			ckernel.GetSeeds().GetSeed(thread.in_grid_idx + path_depth));

		if (path_depth >= ckernel.GetRenderConfig().GetTracing().GetMaxDepth())
		{
			// generate camera ray
			SceneRay camera_ray;
			camera.GenerateRay(
				camera_ray,
				thread.in_grid,
				rng);
			camera_ray.material = &world->material;

			camera.GetTracingStates().SetRay(thread.in_grid, camera_ray);
			camera.GetTracingStates().SetPathDepth(thread.in_grid, 0u);
		}
	}
	__global__ void SwapPathIndexing(
		World* const world,
		const uint32_t camera_idx)
	{
		Camera& camera = world->cameras[camera_idx];
		camera.SwapPathIndexing();
	}


	__device__ vec3f TraceRay(
		ConstantKernel& ckernel,
		const World& world,
		TracingState& tracing_state,
		RayIntersection& intersection,
		RNG& rng)
	{
		// find closest intersection with the world
		const bool any_hit = world.ClosestIntersection(intersection, rng);

		// fetch color and emission at given point on material surface
		intersection.color =
			intersection.surface_material->GetOpacityColor(intersection.texcrd);
		intersection.emission =
			intersection.surface_material->GetEmission(intersection.texcrd);

		if (intersection.emission > 0.0f)
		{	// intersection with emitting object

			tracing_state.final_color +=
				intersection.ray.color *
				intersection.color *
				intersection.emission;
		}

		if (!any_hit)
		{	// nothing has been hit - terminate path

			tracing_state.EndPath();
			return vec3f(1.0f);
		}
		++tracing_state.path_depth;


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


		// Fetch metalness  and roughness from surface material
		// (needed for BRDF and next even estimation)
		intersection.metalness =
			intersection.surface_material->GetMetalness(intersection.texcrd);
		intersection.roughness =
			intersection.surface_material->GetRoughness(intersection.texcrd);

		// calculate fresnel and reflectance ratio
		// (for BRDF and next ray generation)
		intersection.fresnel = FresnelSpecularRatio(
			intersection.mapped_normal,
			intersection.ray.direction,
			intersection.ray.material->GetIOR(),
			intersection.behind_material->GetIOR());
		intersection.reflectance = Lerp(intersection.fresnel, 1.0f, intersection.metalness);

		// find intersection point 
		// (needed for direct sampling and next ray generation)
		intersection.point =
			intersection.ray.origin +
			intersection.ray.direction *
			intersection.ray.near_far.y;

		// sample direction (importance sampling)
		// (for next ray generation and direct light sampling (MIS))
		const vec3f sample_direction = intersection.surface_material->SampleDirection(intersection, rng);

		// Direct sampling
		if (intersection.surface_material->SampleDirect(intersection) &&
			world.SampleDirect(ckernel))
		{
			// sample direct light
			const ColorF direct_illumination = DirectIllumination(ckernel, world, intersection, sample_direction, rng);

			// add direct light
			tracing_state.final_color +=
				direct_illumination * // incoming radiance from lights
				intersection.ray.color * // ray color mask
				Lerp(ColorF(1.0f), intersection.color, intersection.metalness); // metalic factor
		}

		return sample_direction;
	}

	__device__ ColorF SpotLightSampling(
		ConstantKernel& ckernel,
		const World& world,
		const RayIntersection& intersection,
		const vec3f& vS,
		const float vS_pdf,
		RNG& rng)
	{
		const uint32_t light_count = world.spot_lights.GetCount();
		const uint32_t sample_count = ckernel.GetRenderConfig().GetLightSampling().GetSpotLight();
		if (light_count == 0u || sample_count == 0u)
			return ColorF(0.0f);


		ColorF total_light(0.0f);
		for (uint32_t i = 0u; i < sample_count; ++i)
		{
			const SpotLight& light = world.spot_lights[uint32_t(rng.UnsignedUniform() * light_count)];

			// sample light
			float Se = 0.0f;
			const vec3f vPL = light.SampleDirection(
				intersection.point,
				vS, Se,
				rng);
			const float dPL = vPL.Length();

			const float brdf = intersection.surface_material->BRDF(intersection, vPL / dPL);
			if (brdf < 1.0e-4f) continue;
			const ColorF brdf_color = intersection.surface_material->BRDFColor(intersection);
			const float solid_angle = light.SolidAngle(dPL);
			const float sctr_factor = cui_expf(-dPL * intersection.ray.material->GetScattering());

			// beam illumination
			const float beamIllum = light.BeamIllumination(vPL);
			if (beamIllum < 1.0e-4f)
				continue;

			// calculate radiance at P
			const float L_pdf = 1.0f / solid_angle;
			const float vSw = vS_pdf / (vS_pdf + L_pdf);
			const float Lw = 1.0f - vSw;
			const float Le = light.GetEmission() * solid_angle * brdf;
			const float radiance = (Le * Lw + Se * vSw) * sctr_factor * beamIllum;
			if (radiance < 1.0e-4f) continue;	// unimportant light contribution

			// cast shadow ray and calculate color contribution
			const RangedRay shadowRay(intersection.point + intersection.surface_normal * 0.001f, vPL, vec2f(0.0f, dPL));
			const ColorF V_PL = world.AnyIntersection(shadowRay);
			total_light +=
				light.GetColor() *
				brdf_color *
				radiance *
				V_PL * V_PL.alpha;
		}

		const float pdf = sample_count / float(light_count);
		return total_light / pdf;
	}
	__device__ ColorF DirectLightSampling(
		ConstantKernel& ckernel,
		const World& world,
		const RayIntersection& intersection,
		const vec3f& vS,
		const float vS_pdf,
		RNG& rng)
	{
		const uint32_t light_count = world.direct_lights.GetCount();
		const uint32_t sample_count = ckernel.GetRenderConfig().GetLightSampling().GetDirectLight();
		if (light_count == 0u || sample_count == 0u)
			return ColorF(0.0f);

		ColorF total_light(0.0f);
		for (uint32_t i = 0u; i < sample_count; ++i)
		{
			const auto& light = world.direct_lights[uint32_t(rng.UnsignedUniform() * light_count)];

			// sample light
			float Se = 0.0f;
			const vec3f vPL = light.SampleDirection(
				vS, Se,
				rng);

			const float brdf = intersection.surface_material->BRDF(intersection, vPL.Normalized());
			const ColorF brdf_color = intersection.surface_material->BRDFColor(intersection);
			const float solid_angle = light.SolidAngle();

			// calculate radiance at P
			const float L_pdf = 1.0f / solid_angle;
			const float vSw = vS_pdf / (vS_pdf + L_pdf);
			const float Lw = 1.0f - vSw;
			const float Le = light.GetEmission() * solid_angle * brdf;
			const float radiance = (Le * Lw + Se * vSw);
			if (radiance < 1.0e-4f) continue;	// unimportant light contribution

			// cast shadow ray and calculate color contribution
			const RangedRay shadowRay(intersection.point + intersection.surface_normal * 0.0001f, vPL);
			const ColorF V_PL = world.AnyIntersection(shadowRay);
			total_light +=
				light.GetColor() *
				brdf_color *
				radiance *
				V_PL * V_PL.alpha;
		}

		const float pdf = sample_count / float(light_count);
		return total_light / pdf;
	}
	__device__ ColorF DirectIllumination(
		ConstantKernel& ckernel,
		const World& world,
		const RayIntersection& intersection,
		const vec3f& vS,
		RNG& rng)
	{
		const float vS_pdf = intersection.surface_material->BRDF(intersection, vS);

		return
			SpotLightSampling(ckernel, world, intersection, vS, vS_pdf, rng) +
			DirectLightSampling(ckernel, world, intersection, vS, vS_pdf, rng);
	}



	// kernels in shared memory:
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
}