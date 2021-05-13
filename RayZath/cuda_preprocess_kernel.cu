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
				if (!camera->Exist()) return;

				// calculate thread position
				const uint32_t thread_x = blockIdx.x * blockDim.x + threadIdx.x;
				const uint32_t thread_y = blockIdx.y * blockDim.y + threadIdx.y;
				if (thread_x >= camera->GetWidth() || thread_y >= camera->GetHeight()) return;

				camera->EmptyImageBuffer().SetValue(ColorF(0.0f), thread_x, thread_y);
				camera->EmptyPassesBuffer().SetValue(1u, thread_x, thread_y);
				camera->SampleDepthBuffer().SetValue(1.0e+30f, thread_x, thread_y);
			}
			__global__ void SpacialReprojection(
				CudaWorld* const world,
				const int camera_id)
			{
				CudaCamera* const camera = &world->cameras[camera_id];
				if (!camera->Exist()) return;

				// calculate thread position
				const uint32_t thread_x = blockIdx.x * blockDim.x + threadIdx.x;
				const uint32_t thread_y = blockIdx.y * blockDim.y + threadIdx.y;
				if (thread_x >= camera->GetWidth() || thread_y >= camera->GetHeight()) return;

				camera->Reproject(thread_x, thread_y);
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