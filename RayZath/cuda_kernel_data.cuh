#ifndef CUDA_KERNEL_DATA_CUH
#define CUDA_KERNEL_DATA_CUH

#include "cuda_render_parts.cuh"

namespace RayZath::Cuda::Kernel
{
	struct Seeds
	{
		static constexpr uint32_t s_count = 0x100;
		float m_seeds[s_count];

		__host__ void Reconstruct();

		__device__ float GetSeed(const uint32_t id) const
		{
			return m_seeds[id % s_count];
		}
	};
	struct RenderConfig
	{
	public:
		struct LightSampling
		{
		private:
			uint8_t m_spot_light, m_direct_light;

		public:
			__host__ LightSampling& operator=(const RayZath::Engine::LightSampling& light_sampling);

			uint8_t __device__ GetSpotLight() const
			{
				return m_spot_light;
			}
			uint8_t __device__ GetDirectLight() const
			{
				return m_direct_light;
			}
		};
	private:
		LightSampling m_light_sampling;

	public:
		__host__ RenderConfig& operator=(const RayZath::Engine::RenderConfig& render_config);

	public:
		__device__ const LightSampling& GetLightSampling() const
		{
			return m_light_sampling;
		}
	};

	struct ConstantKernel
	{
	private:
		Seeds m_seeds;
		RenderConfig m_render_config;

	public:
		__host__ void Reconstruct(const RayZath::Engine::RenderConfig& render_config);

		__device__ const Seeds& GetSeeds() const
		{
			return m_seeds;
		}
		__device__ const RenderConfig& GetRenderConfig() const
		{
			return m_render_config;
		}
	};
	class GlobalKernel
	{
	private:
		uint32_t m_render_idx;


	public:
		__host__ GlobalKernel();
		__host__ GlobalKernel(const GlobalKernel&) = delete;
		__host__ GlobalKernel(GlobalKernel&&) = delete;

	public:
		__host__ void Reconstruct(
			uint32_t render_idx,
			cudaStream_t& stream);

		__device__ uint32_t GetRenderIdx() const
		{
			return m_render_idx;
		}
	};

	extern __constant__ ConstantKernel const_kernel[2];

	__host__ void CopyConstantKernel(
		const ConstantKernel* hCudaConstantKernel,
		const uint32_t& update_idx,
		cudaStream_t& stream);
}

#endif