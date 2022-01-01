#include "cuda_render_kernel.cuh"

#include "math_constants.h"

namespace RayZath::Cuda::Kernel
{
	// ~~~~~~~~ Memory Management ~~~~~~~~
	__constant__ ConstantKernel const_kernel[2];
	__device__ const ConstantKernel* ckernel;

	__host__ void CopyToConstantMemory(
		const ConstantKernel* hCudaConstantKernel,
		const uint32_t& update_idx,
		cudaStream_t& stream)
	{
		CudaErrorCheck(cudaMemcpyToSymbolAsync(
			(const void*)const_kernel, hCudaConstantKernel,
			sizeof(ConstantKernel), update_idx * sizeof(ConstantKernel),
			cudaMemcpyKind::cudaMemcpyHostToDevice, stream));
		CudaErrorCheck(cudaStreamSynchronize(stream));
	}


	__global__ void LaunchFirstPass(
		GlobalKernel* const global_kernel,
		World* world,
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
		GlobalKernel* const kernel = global_kernel;
		ckernel = &const_kernel[kernel->GetRenderIdx()];

		// create thread object
		FullThread thread;

		// get camera and clamp working threads
		Camera* const camera = &world->cameras[camera_idx];
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
		GlobalKernel* const global_kernel,
		World* world,
		const int camera_idx)
	{
		// get kernels
		GlobalKernel* const kernel = global_kernel;
		ckernel = &const_kernel[kernel->GetRenderIdx()];

		FullThread thread;

		// get camera and clamp working threads
		Camera* const camera = &world->cameras[camera_idx];
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
		const World& World,
		Camera& camera,
		TracingPath& tracing_path,
		RayIntersection& intersection,
		RNG& rng)
	{
		// trace ray through scene
		TraceRay(World, tracing_path, intersection, rng);

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
			intersection.ray.color * intersection.color,
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
				intersection.ray.color * intersection.color,
				metalness_ratio);

		}
	}
	__device__ void RenderCumulativePass(
		const World& World,
		Camera& camera,
		TracingPath& tracing_path,
		RayIntersection& intersection,
		RNG& rng)
	{
		do
		{
			// trace ray through scene
			TraceRay(World, tracing_path, intersection, rng);

			if (!tracing_path.NextNodeAvailable())
				return;

			// generate next ray
			const float metalness_ratio =
				intersection.surface_material->GenerateNextRay(
					intersection,
					rng);

			// multiply color mask by surface color according to material metalness
			intersection.ray.color.Blend(
				intersection.ray.color * intersection.color,
				metalness_ratio);

		} while (tracing_path.FindNextNodeToTrace());
	}

	__device__ void TraceRay(
		const World& world,
		TracingPath& tracing_path,
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

			tracing_path.finalColor +=
				intersection.ray.color *
				intersection.color *
				intersection.emission;
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


		// Fetch metalness  and roughness from surface material
		// (needed for BRDF and next even estimation)
		intersection.metalness =
			intersection.surface_material->GetMetalness(intersection.texcrd);
		intersection.roughness =
			intersection.surface_material->GetRoughness(intersection.texcrd);

		// calculate reflectance ratio
		// (for BRDF and next ray generation)
		intersection.reflectance =
			Lerp(FresnelSpecularRatio(
				intersection.mapped_normal,
				intersection.ray.direction,
				intersection.ray.material->GetIOR(),
				intersection.behind_material->GetIOR()),
				1.0f,
				intersection.metalness);

		// find intersection point 
		// (needed for direct sampling and next ray generation)
		intersection.point =
			intersection.ray.origin +
			intersection.ray.direction *
			intersection.ray.near_far.y;


		// Direct sampling
		if (intersection.surface_material->SampleDirect(intersection))
		{
			// sample direct light
			const ColorF direct_light = DirectSampling(world, intersection, rng);

			// add direct light
			tracing_path.finalColor +=
				direct_light * // incoming radiance from lights
				intersection.ray.color * // ray color mask
				Lerp(ColorF(1.0f), intersection.color, intersection.metalness); // metalic factor
		}
	}

	__device__ Color<float> DirectSampling(
		const World& world,
		RayIntersection& intersection,
		RNG& rng)
	{
		// Legend:
		// L - position of current light
		// P - point of intersetion
		// vN - surface normal

		ColorF total_light(0.0f);
		ColorF total_type_light(0.0f);

		// [>] PointLights
		for (uint32_t i = 0u; i < ckernel->GetRenderConfig().GetLightSampling().GetPointLight(); ++i)
		{
			const PointLight& light = world.point_lights[
				uint32_t(rng.UnsignedUniform() * float(world.point_lights.GetCount()))];

			// sample light
			const vec3f vPL = light.SampleDirection(
				intersection.point,
				rng);
			const float dPL = vPL.Length();

			const ColorF brdf = intersection.surface_material->BRDF(intersection, vPL / dPL);
			const float solid_angle = light.SolidAngle(dPL);
			const float sctr_factor = cui_expf(-dPL * intersection.ray.material->GetScattering());

			// calculate radiance at P
			const ColorF radianceP = brdf * light.material.GetEmission() * solid_angle * sctr_factor;
			if (radianceP.red + radianceP.green + radianceP.blue < 3.0e-4f) continue;	// unimportant light contribution

			// cast shadow ray and calculate color contribution
			const Ray shadowRay(intersection.point + intersection.surface_normal * 0.0001f, vPL, vec2f(0.0f, dPL));
			const ColorF shadow_color = world.AnyIntersection(shadowRay);
			total_type_light += light.material.GetColor() * shadow_color * shadow_color.alpha * radianceP;
		}
		total_light += total_type_light /
			(float(ckernel->GetRenderConfig().GetLightSampling().GetPointLight()) /
				world.point_lights.GetCount());


		// [>] SpotLights
		total_type_light = ColorF(0.0f);
		for (uint32_t i = 0u; i < world.spot_lights.GetCount(); ++i)
		{
			const SpotLight& light = world.spot_lights[i];

			// sample light
			const vec3f vPL = light.SampleDirection(
				intersection.point,
				rng);
			const float dPL = vPL.Length();

			const ColorF brdf = intersection.surface_material->BRDF(intersection, vPL / dPL);
			const float solid_angle = light.SolidAngle(dPL);
			const float sctr_factor = cui_expf(-dPL * intersection.ray.material->GetScattering());

			// beam illumination
			float beamIllum = 1.0f;
			const float LP_dot_D = vec3f::Similarity(-vPL, light.direction);
			if (LP_dot_D < light.cos_angle) beamIllum = 0.0f;
			else beamIllum = 1.0f;

			// calculate radiance at P
			const ColorF radianceP = brdf * light.material.GetEmission() * solid_angle * sctr_factor * beamIllum;
			if (radianceP.red + radianceP.green + radianceP.blue < 3.0e-4f) continue;	// unimportant light contribution

			// cast shadow ray and calculate color contribution
			const Ray shadowRay(intersection.point + intersection.surface_normal * 0.001f, vPL, vec2f(0.0f, dPL));
			const ColorF shadow_color = world.AnyIntersection(shadowRay);
			total_type_light += light.material.GetColor() * shadow_color * shadow_color.alpha * radianceP;
		}
		total_light += total_type_light /
			(float(ckernel->GetRenderConfig().GetLightSampling().GetSpotLight()) /
				world.point_lights.GetCount());


		// [>] DirectLights
		total_type_light = ColorF(0.0f);
		for (uint32_t i = 0u; i < world.direct_lights.GetCount(); ++i)
		{
			const DirectLight& light = world.direct_lights[i];

			// sample light
			const vec3f vPL = light.SampleDirection(rng);

			const ColorF brdf = intersection.surface_material->BRDF(intersection, vPL.Normalized());
			const float solid_angle = light.SolidAngle();

			// calculate radiance at P
			const ColorF radianceP = brdf * light.material.GetEmission() * solid_angle;
			if (radianceP.red + radianceP.green + radianceP.blue < 3.0e-4f) continue;	// unimportant light contribution

			// cast shadow ray and calculate color contribution
			const Ray shadowRay(intersection.point + intersection.surface_normal * 0.0001f, vPL);
			const ColorF shadow_color = world.AnyIntersection(shadowRay);
			total_type_light += light.material.GetColor() * shadow_color * shadow_color.alpha * radianceP;
		}
		total_light += total_type_light /
			(float(ckernel->GetRenderConfig().GetLightSampling().GetDirectLight()) /
				world.point_lights.GetCount());

		return total_light;
	}
}