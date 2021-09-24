#include "cuda_preprocess_kernel.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		namespace CudaKernel
		{
			__global__ void DepthBufferReset(
				CudaWorld* const world,
				const int camera_id)
			{
				CudaCamera* const camera = &world->cameras[camera_id];

				// calculate thread position
				GridThread thread;
				if (thread.in_grid.x >= camera->GetWidth() || thread.in_grid.y >= camera->GetHeight()) return;

				camera->EmptyPassesBuffer().SetValue(thread.in_grid, 1u);
			}
			__global__ void CudaCameraUpdateSamplesNumber(
				CudaWorld* const world,
				const int camera_id,
				bool reset_flag)
			{
				CudaCamera* const camera = &world->cameras[camera_id];

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
	}
}