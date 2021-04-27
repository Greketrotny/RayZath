#include "cuda_preprocess_kernel.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		namespace CudaKernel
		{
			__global__ void CudaCameraSampleReset(
				CudaWorld* const world,
				const int camera_id)
			{
				CudaCamera* const camera = &world->cameras[camera_id];
				if (!camera->Exist()) return;

				// calculate thread position
				const uint32_t thread_x = blockIdx.x * blockDim.x + threadIdx.x;
				const uint32_t thread_y = blockIdx.y * blockDim.y + threadIdx.y;
				if (thread_x >= camera->GetWidth() || thread_y >= camera->GetHeight()) return;

				// reset sample buffer
				camera->SetSamplePixel(Color<float>(0.0f, 0.0f, 0.0f, FLT_EPSILON), thread_x, thread_y);

				// TODO: reset tracing paths
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
					camera->GetInvPassesCount() = 1.0f;
				}
				else
				{
					camera->GetPassesCount() += 1u;
					camera->GetInvPassesCount() = 1.0f / float(camera->GetPassesCount());
				}
			}
		}
	}
}