#include "cuda_postprocess_kernel.cuh"

namespace RayZath::Cuda::Kernel
{
	__global__ void SpacialReprojection(
		World* const world,
		const uint32_t camera_idx)
	{
		const GridThread thread;

		Camera& camera = world->cameras[camera_idx];
		if (thread.grid_pos.x >= camera.GetWidth() ||
			thread.grid_pos.y >= camera.GetHeight()) return;

		camera.Reproject(thread.grid_pos);
	}


	__device__ __inline__ ColorF ToneMap_ACES(const ColorF& v)
	{
		constexpr float a = 2.51f;
		constexpr float b = 0.03f;
		constexpr float c = 2.43f;
		constexpr float d = 0.59f;
		constexpr float e = 0.14f;

		const ColorF r = ((v * (v * a + ColorF(b)) / ((v * (v * c + ColorF(d)) + ColorF(e)))));
		return ColorF(
			__saturatef(r.red),
			__saturatef(r.green),
			__saturatef(r.blue),
			__saturatef(r.alpha));
	}
	__device__ __inline__ ColorF ToneMap_Hyper(const ColorF& v)
	{
		return v / (v + ColorF(1.0f));
	}


	__device__ __inline__ void ComputeFinalColor(
		const GridThread& thread,
		Camera& camera)
	{
		// average sample color by dividing by number of samples
		ColorF pixel = camera.CurrentImageBuffer().GetValue(thread.grid_pos);
		pixel /= pixel.alpha;

		pixel *= camera.GetApertureArea();
		pixel *= camera.GetExposureTime();
		pixel *= 1.0e5f;	// camera matrix sensitivity.		
		pixel = ToneMap_Hyper(pixel);

		camera.FinalImageBuffer().SetValue(
			thread.grid_pos,
			ColorU(
				pixel.red * 255.0f,
				pixel.green * 255.0f,
				pixel.blue * 255.0f,
				255u));
	}
	__device__ __inline__ void ComputeDepth(
		const GridThread& thread,
		Camera& camera)
	{
		camera.FinalDepthBuffer().SetValue(
			thread.grid_pos,
			camera.CurrentDepthBuffer().GetValue(thread.grid_pos));
	}
	__global__ void FirstToneMap(
		World* const world,
		const uint32_t camera_idx)
	{
		const GridThread thread;

		// calculate thread position
		Camera& camera = world->cameras[camera_idx];
		if (thread.grid_pos.x >= camera.GetWidth() ||
			thread.grid_pos.y >= camera.GetHeight()) return;

		ComputeFinalColor(thread, camera);
		ComputeDepth(thread, camera);		
	}
	__global__ void ToneMap(
		World* const world,
		const uint32_t camera_idx)
	{
		const GridThread thread;

		// calculate thread position
		Camera& camera = world->cameras[camera_idx];
		if (thread.grid_pos.x >= camera.GetWidth() ||
			thread.grid_pos.y >= camera.GetHeight()) return;

		ComputeFinalColor(thread, camera);
	}


	__global__ void PassUpdate(
		World* const world,
		const uint32_t camera_idx)
	{
		Camera& camera = world->cameras[camera_idx];

		camera.SetRenderPassCount(camera.GetRenderPassCount() + 1u);
		camera.SetResultPassCount(camera.GetRenderPassCount());
		camera.SetResultRayCount(camera.GetRenderRayCount());
	}

	/*
	__global__ void IrradianceReduction(
		GlobalKernel* const global_kernel,
		World* const world,
		const int camera_id)
	{
		Camera& camera = world->cameras[camera_id];

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
}