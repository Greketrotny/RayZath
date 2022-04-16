#include "cuda_render_kernel.cuh"

#include "math_constants.h"

namespace RayZath::Cuda::Kernel
{
	__global__ void RenderFirstPass(
		GlobalKernel* const global_kernel,
		World* const world,
		const uint32_t camera_idx)
	{
		const GridThread thread;

		// get camera and clamp working threads
		Camera& camera = world->cameras[camera_idx];
		if (thread.grid_pos.x >= camera.GetWidth() ||
			thread.grid_pos.y >= camera.GetHeight()) return;

		// get kernels
		GlobalKernel& gkernel = *global_kernel;
		ConstantKernel& ckernel = const_kernel[gkernel.GetRenderIdx()];

		// create RNG
		RNG rng(
			vec2f(
				thread.grid_pos.x / float(camera.GetWidth()),
				thread.grid_pos.y / float(camera.GetHeight())),
			ckernel.GetSeeds().GetSeed(thread.grid_idx));

		// create intersection object
		SceneRay ray = camera.GetTracingStates().GetRay(thread.grid_pos);
		ray.near_far = camera.GetNearFar();

		// trace ray through scene
		TracingState tracing_state(ColorF(0.0f), 0u);
		const TracingResult result = TraceRay(ckernel, *world, tracing_state, ray, rng);
		const bool path_continues = tracing_state.path_depth < ckernel.GetRenderConfig().GetTracing().GetMaxDepth();

		// set depth
		camera.CurrentDepthBuffer().SetValue(thread.grid_pos, ray.near_far.y);
		// set intersection point
		camera.SpaceBuffer().SetValue(thread.grid_pos, 
			ray.origin + ray.direction * ray.near_far.y);
		// set color value
		tracing_state.final_color.alpha = float(!path_continues);
		camera.CurrentImageBuffer().SetValue(
			thread.grid_pos,
			tracing_state.final_color);

		if (path_continues)
		{
			result.RepositionRay(ray);
		}
		else
		{
			// generate camera ray
			camera.GenerateRay(ray, thread.grid_pos, rng);
			ray.material = &world->material;
			ray.color = ColorF(1.0f);
		}

		camera.GetTracingStates().SetRay(thread.grid_pos, ray);
		camera.GetTracingStates().SetPathDepth(
			thread.grid_pos,
			path_continues ? tracing_state.path_depth : 0u);
	}
	__global__ void RenderCumulativePass(
		GlobalKernel* const global_kernel,
		World* const world,
		const uint32_t camera_idx)
	{
		const GridThread thread;

		Camera& camera = world->cameras[camera_idx];
		if (thread.grid_pos.x >= camera.GetWidth() ||
			thread.grid_pos.y >= camera.GetHeight()) return;

		// get kernels
		GlobalKernel& gkernel = *global_kernel;
		ConstantKernel& ckernel = const_kernel[gkernel.GetRenderIdx()];

		TracingState tracing_state(
			ColorF(0.0f),
			camera.GetTracingStates().GetPathDepth(thread.grid_pos));

		RNG rng(
			vec2f(
				thread.grid_pos.x / float(camera.GetWidth()),
				thread.grid_pos.y / float(camera.GetHeight())),
			ckernel.GetSeeds().GetSeed(thread.grid_idx + tracing_state.path_depth));


		// trace ray through scene
		SceneRay ray = camera.GetTracingStates().GetRay(thread.grid_pos);
		const TracingResult result = TraceRay(ckernel, *world, tracing_state, ray, rng);
		const bool path_continues = tracing_state.path_depth < ckernel.GetRenderConfig().GetTracing().GetMaxDepth();

		// append additional light contribution passing along traced ray
		ColorF sample = camera.CurrentImageBuffer().GetValue(thread.grid_pos);
		sample.red += tracing_state.final_color.red;
		sample.green += tracing_state.final_color.green;
		sample.blue += tracing_state.final_color.blue;
		sample.alpha += float(!path_continues);
		camera.CurrentImageBuffer().SetValue(thread.grid_pos, sample);

		if (path_continues)
		{
			result.RepositionRay(ray);
		}
		else
		{
			// generate camera ray
			camera.GenerateRay(ray, thread.grid_pos, rng);
			ray.material = &world->material;
			ray.color = ColorF(1.0f);
		}

		camera.GetTracingStates().SetRay(thread.grid_pos, ray);
		camera.GetTracingStates().SetPathDepth(thread.grid_pos, path_continues ? tracing_state.path_depth : 0u);
	}
	__global__ void SegmentUpdate(
		World* const world,
		const uint32_t camera_idx)
	{
		Camera& camera = world->cameras[camera_idx];

		camera.SetRenderRayCount(camera.GetRenderRayCount() + uint64_t(camera.GetWidth()) * uint64_t(camera.GetHeight()));
	}


	__device__ TracingResult TraceRay(
		ConstantKernel& ckernel,
		const World& world,
		TracingState& tracing_state,
		SceneRay& ray,
		RNG& rng)
	{
		// find closest intersection with the world
		SurfaceProperties surface(&world.material);
		const bool any_hit = world.ClosestIntersection(ray, surface, rng);

		// fetch color and emission at given point on material surface
		surface.color = surface.surface_material->GetOpacityColor(surface.texcrd);
		surface.emission = surface.surface_material->GetEmission(surface.texcrd);

		// apply Beer's law
		/*
		 P0 - light energy in front of an object
		 P - light energy after going through an object
		 A - absorbance

		 e - material absorbance (constant)
		 b - distance traveled in an object
		 c - molar concentration (constant)

		 A = 10 ^ -(e * b * c)
		 P = P0 * A
		*/
		ray.color *=
			ray.material->GetOpacityColor() *
			cui_powf(ray.material->GetOpacityColor().alpha, ray.near_far.y);


		if (surface.emission > 0.0f)
		{	// intersection with emitting object

			tracing_state.final_color +=
				ray.color *
				surface.color *
				surface.emission;
		}

		if (!any_hit)
		{	// nothing has been hit - terminate path

			tracing_state.EndPath();
			return TracingResult{};
		}
		++tracing_state.path_depth;


		// Fetch metalness  and roughness from surface material (for BRDF and next even estimation)
		surface.metalness = surface.surface_material->GetMetalness(surface.texcrd);
		surface.roughness = surface.surface_material->GetRoughness(surface.texcrd);

		// calculate fresnel and reflectance ratio (for BRDF and next ray generation)
		surface.fresnel = FresnelSpecularRatio(
			surface.mapped_normal,
			ray.direction,
			ray.material->GetIOR(),
			surface.behind_material->GetIOR(),
			surface.refraction_factors);
		surface.reflectance = Lerp(surface.fresnel, 1.0f, surface.metalness);

		TracingResult result;
		// sample direction (importance sampling) (for next ray generation and direct light sampling (MIS))
		result.next_direction = surface.surface_material->SampleDirection(ray, surface, rng);
		// find intersection point (for direct sampling and next ray generation)
		result.point = 
			ray.origin + ray.direction * ray.near_far.y +	// origin + ray vector
			surface.normal * (0.0001f * ray.near_far.y);	// normal nodge

		// Direct sampling
		if (surface.surface_material->SampleDirect(surface) &&
			world.SampleDirect(ckernel))
		{
			// sample direct light
			const ColorF direct_illumination = DirectIllumination(
				ckernel, world, 
				ray, result, surface,
				rng);

			// add direct light
			tracing_state.final_color +=
				direct_illumination * // incoming radiance from lights
				ray.color * // ray color mask
				Lerp(ColorF(1.0f), surface.color, surface.metalness); // metalic factor
		}

		ray.color.Blend(ray.color * surface.color, surface.tint_factor);
		return result;
	}

	__device__ ColorF SpotLightSampling(
		ConstantKernel& ckernel,
		const World& world,
		const SceneRay& ray,
		const TracingResult& result,
		const SurfaceProperties& surface,
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
			const vec3f vPL = light.SampleDirection(result.point, result.next_direction, Se, rng);
			const float dPL = vPL.Length();

			const float brdf = surface.surface_material->BRDF(ray, surface, vPL / dPL);
			if (brdf < 1.0e-4f) continue;
			const ColorF brdf_color = surface.surface_material->BRDFColor(surface);
			const float solid_angle = light.SolidAngle(dPL);
			const float sctr_factor = cui_expf(-dPL * ray.material->GetScattering());

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
			const RangedRay shadowRay(result.point, vPL, vec2f(0.0f, dPL));
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
		const SceneRay& ray,
		const TracingResult& result,
		const SurfaceProperties& surface,
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
			const vec3f vPL = light.SampleDirection(result.next_direction, Se, rng);

			const float brdf = surface.surface_material->BRDF(ray, surface, vPL.Normalized());
			const ColorF brdf_color = surface.surface_material->BRDFColor(surface);
			const float solid_angle = light.SolidAngle();

			// calculate radiance at P
			const float L_pdf = 1.0f / solid_angle;
			const float vSw = vS_pdf / (vS_pdf + L_pdf);
			const float Lw = 1.0f - vSw;
			const float Le = light.GetEmission() * solid_angle * brdf;
			const float radiance = (Le * Lw + Se * vSw);
			if (radiance < 1.0e-4f) continue;	// unimportant light contribution

			// cast shadow ray and calculate color contribution
			const RangedRay shadowRay(result.point, vPL);
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
		const SceneRay& ray,
		const TracingResult& result,
		const SurfaceProperties& surface,
		RNG& rng)
	{
		const float vS_pdf = surface.surface_material->BRDF(ray, surface, result.next_direction);

		return
			SpotLightSampling(ckernel, world, ray, result, surface, vS_pdf, rng) +
			DirectLightSampling(ckernel, world, ray, result, surface, vS_pdf, rng);
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