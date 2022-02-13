#include "cuda_preprocess_kernel.cuh"

namespace RayZath::Cuda::Kernel
{
	__global__ void SwapBuffers(
		World* const world,
		const uint8_t camera_idx)
	{
		Camera& camera = world->cameras[camera_idx];
		camera.SwapImageBuffers();
		camera.GetPassesCount() = 1u;
	}
	__global__ void UpdatePassesCount(
		World* const world,
		const uint8_t camera_idx)
	{
		Camera& camera = world->cameras[camera_idx];
		camera.GetPassesCount() += 1u;
	}

	__global__ void GenerateCameraRay(
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

		// create RNG
		RNG rng(
			vec2f(
				thread.in_grid.x / float(camera.GetWidth()),
				thread.in_grid.y / float(camera.GetHeight())),
			ckernel.GetSeeds().GetSeed(thread.in_grid_idx));


		// generate camera ray
		SceneRay camera_ray;
		camera.GenerateSimpleRay(
			camera_ray,
			thread,
			rng);
		camera_ray.material = &world->material;

		camera.GetTracingStates().SetRay(camera_ray, thread.in_grid);
	}
}