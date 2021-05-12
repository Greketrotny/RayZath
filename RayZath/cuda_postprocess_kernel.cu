#include "cuda_postprocess_kernel.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		namespace CudaKernel
		{
			// [>] Tone mapping
			/*__global__ void IrradianceReduction(
				CudaGlobalKernel* const global_kernel,
				CudaWorld* const world,
				const int camera_id)
			{
				CudaCamera& camera = world->cameras[camera_id];

				uint32_t thread_x = blockIdx.x * blockDim.x + threadIdx.x;
				uint32_t thread_y = blockIdx.y * blockDim.y + threadIdx.y;
				uint32_t thread_in_block = threadIdx.y * blockDim.x + threadIdx.x;
				if (thread_x >= camera.GetWidth() || thread_y >= camera.GetHeight()) return;

				extern __shared__ ColorF block_fragment[];
				block_fragment[thread_in_block] = ColorF(0.0f);


				// [>] Grid reduction
				const float inv_passes_count = 1.0f / float(camera.GetPassesCount());
				for (uint32_t y = threadIdx.y; y < camera.GetHeight(); y += blockDim.y)
				{
					for (uint32_t x = threadIdx.x; x < camera.GetWidth(); x += blockDim.x)
					{
						block_fragment[thread_in_block] += camera.GetSamplePixel(x, y) * inv_passes_count;
					}
				}


				// [>] Block reduction
				// find greatest 2^n smaller or equal to blockDim.y
				uint32_t n = blockDim.y;
				n = n | (n >> 1);
				n = n | (n >> 2);
				n = n | (n >> 4);
				n = n | (n >> 8);
				n = n | (n >> 16);
				++n;
				n >>= 1u;
				__syncthreads();

				// blockDim.y - 2^n reduction
				if (threadIdx.y >= n)
				{
					block_fragment[(threadIdx.y - n) * blockDim.x + threadIdx.x] +=
						block_fragment[threadIdx.y * blockDim.x + threadIdx.x];
				}
				n >>= 1u;
				__syncthreads();

				// block 2^n reduction
				for (; n > 0u; n >>= 1u)
				{
					if (threadIdx.y < n)
					{
						block_fragment[threadIdx.y * blockDim.x + threadIdx.x] +=
							block_fragment[(threadIdx.y + n) * blockDim.x + threadIdx.x];
					}
				}
				__syncthreads();
				
				// warp reduction
				if (threadIdx.y == 0u)
				{
					for (uint32_t i = warpSize / 2u; i > 0u; i >>= 1u)
					{
						if (threadIdx.x < i)
						{
							block_fragment[threadIdx.x] +=
								block_fragment[threadIdx.x + i];
						}
					}
				}
				__syncthreads();

				if (thread_in_block == 0u)
				{
					camera.something_sumed = block_fragment[0u];
				}
			}*/
		

			__device__ __inline__ ColorF HDRtoLDR(const ColorF& v)
			{
				return v / (v + ColorF(1.0f));
			}

			__global__ void ToneMap(
				CudaGlobalKernel* const global_kernel,
				CudaWorld* const world,
				const int camera_id)
			{
				CudaCamera* const camera = &world->cameras[camera_id];

				// calculate thread position
				const uint32_t thread_x = blockIdx.x * blockDim.x + threadIdx.x;
				const uint32_t thread_y = blockIdx.y * blockDim.y + threadIdx.y;
				if (thread_x >= camera->GetWidth() || thread_y >= camera->GetHeight()) return;


				// [>] Calculate pixel color
				// average sample color by dividing by number of samples
				Color<float> pixel =
					camera->SampleImageBuffer().GetValue(thread_x, thread_y);
				pixel /= float(camera->PassesBuffer().GetValue(thread_x, thread_y));

				pixel *= CUDART_PI_F * camera->GetAperture() * camera->GetAperture();
				pixel *= camera->GetExposureTime();
				pixel *= 1.0e5f;	// camera matrix sensitivity.

				pixel = HDRtoLDR(pixel);

				camera->FinalImageBuffer(global_kernel->GetRenderIdx()).SetValue(
					Color<unsigned char>(
						pixel.red * 255.0f,
						pixel.green * 255.0f,
						pixel.blue * 255.0f,
						255u),
					thread_x, thread_y);
				

				// [>] Calculate depth
				const float depth = camera->SampleDepthBuffer().GetValue(thread_x, thread_y) / camera->GetPassesCount();
				camera->FinalDepthBuffer(global_kernel->GetRenderIdx()).SetValue(depth, thread_x, thread_y);
			}
		}
	}
}