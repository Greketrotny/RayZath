#include "cuda_preprocess_kernel.cuh"

namespace RayZath::Cuda::Kernel
{
	__global__ void PassReset(
		World* const world,
		const uint32_t camera_idx)
	{
		Camera& camera = world->cameras[camera_idx];
		camera.SetRenderPassCount(0u);
		camera.SetRenderRayCount(0u);
	}

	__global__ void GenerateCameraRay(
		GlobalKernel* const global_kernel,
		World* const world,
		const uint32_t camera_idx)
	{
		const GridThread thread;

		Camera& camera = world->cameras[camera_idx];
		if (thread.grid_pos.x >= camera.GetWidth() ||
			thread.grid_pos.y >= camera.GetHeight()) return;

		GlobalKernel& gkernel = *global_kernel;
		ConstantKernel& ckernel = const_kernel[gkernel.GetRenderIdx()];

		// generate camera ray
		SceneRay camera_ray;
		camera.GenerateSimpleRay(camera_ray, thread);
		camera_ray.material = &world->material;
		camera.GetTracingStates().SetRay(thread.grid_pos, camera_ray);
	}
}