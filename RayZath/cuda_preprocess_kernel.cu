#include "cuda_preprocess_kernel.cuh"

namespace RayZath::Cuda::Kernel
{
	__global__ void DepthBufferReset(
		World* const world,
		const int camera_id)
	{
		Camera* const camera = &world->cameras[camera_id];

		// calculate thread position
		GridThread thread;
		if (thread.in_grid.x >= camera->GetWidth() || thread.in_grid.y >= camera->GetHeight()) return;

		camera->EmptyPassesBuffer().SetValue(thread.in_grid, 1u);
	}
	__global__ void CudaCameraUpdateSamplesNumber(
		World* const world,
		const int camera_id,
		bool reset_flag)
	{
		Camera* const camera = &world->cameras[camera_id];

		// passes count
		if (reset_flag)
		{
			camera->GetPassesCount() = 1u;
			camera->SwapImageBuffers();
		}
		else
		{
			camera->GetPassesCount() += 1u;
		}
	}

	// -- new pipeline
	__global__ void GenerateCameraRay(
		GlobalKernel* const global_kernel,
		World* const world,
		const uint32_t camera_id)
	{
		Camera& camera = world->cameras[camera_id];
		GridThread thread;
		if (thread.in_grid.x >= camera.GetWidth() ||
			thread.in_grid.y >= camera.GetHeight()) return;

		GlobalKernel* const kernel = global_kernel;
		ConstantKernel* ckernel = &const_kernel[kernel->GetRenderIdx()];

		// create RNG
		RNG rng(
			vec2f(
				thread.in_grid.x / float(camera.GetWidth()),
				thread.in_grid.y / float(camera.GetHeight())),
			ckernel->GetSeeds().GetSeed(thread.in_grid_idx));


		// generate camera ray
		SceneRay camera_ray;
		camera.GenerateSimpleRay(
			camera_ray,
			thread,
			rng);
		camera_ray.material = &world->material;

		camera.GetTracingStates().SetRay(camera_ray, thread);
	}
}