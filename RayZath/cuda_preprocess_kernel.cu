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
}