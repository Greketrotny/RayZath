#include "cuda_preprocess_kernel.cuh"

namespace RayZath::Cuda::Kernel
{
	__global__ void passReset(
		World* const world,
		const uint32_t camera_idx)
	{
		Camera& camera = world->cameras[camera_idx];
		camera.setRenderPassCount(0u);
		camera.setRenderRayCount(0u);
	}

	__global__ void generateCameraRay(
		GlobalKernel* const global_kernel,
		World* const world,
		const uint32_t camera_idx)
	{
		const GridThread thread;

		Camera& camera = world->cameras[camera_idx];
		if (thread.grid_pos.x >= camera.width() ||
			thread.grid_pos.y >= camera.height()) return;

		GlobalKernel& gkernel = *global_kernel;
		ConstantKernel& ckernel = const_kernel[gkernel.renderIdx()];

		// generate camera ray
		SceneRay camera_ray;
		camera.generateSimpleRay(camera_ray, thread);
		camera_ray.material = &world->material;
		camera.getTracingStates().setRay(thread.grid_pos, camera_ray);
	}
}