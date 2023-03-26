#ifndef CUDA_KERNEL_DATA_CUH
#define CUDA_KERNEL_DATA_CUH

#include "cuda_include.hpp"

#include <cstdint>

namespace RayZath::Engine
{
	struct LightSampling;
	struct Tracing;
	struct RenderConfig;
}

namespace RayZath::Cuda::Kernel
{
	struct Seeds
	{
		static constexpr uint32_t s_count = 0x100;
		float m_seeds[s_count];

		__host__ void reconstruct();

		__device__ float getSeed(const uint32_t id) const
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

			__device__ uint8_t spotLight() const
			{
				return m_spot_light;
			}
			__device__ uint8_t directLight() const
			{
				return m_direct_light;
			}
		};
		struct Tracing
		{
		private:
			uint8_t m_max_depth;

		public:
			__host__ Tracing& operator=(const RayZath::Engine::Tracing& tracing);

			__device__ uint8_t maxDepth() const
			{
				return m_max_depth;
			}
		};
	private:
		LightSampling m_light_sampling;
		Tracing m_tracing;

	public:
		__host__ RenderConfig& operator=(const RayZath::Engine::RenderConfig& render_config);

	public:
		__device__ const LightSampling& lightSampling() const
		{
			return m_light_sampling;
		}
		__device__ const Tracing& tracing() const
		{
			return m_tracing;
		}
	};

	struct ConstantKernel
	{
	private:
		Seeds m_seeds;
		RenderConfig m_render_config;

	public:
		__host__ void reconstruct(const RayZath::Engine::RenderConfig& render_config);

		__device__ const Seeds& seeds() const
		{
			return m_seeds;
		}
		__device__ const RenderConfig& renderConfig() const
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
		__host__ void reconstruct(
			uint32_t render_idx,
			cudaStream_t& stream);

		__device__ uint32_t renderIdx() const
		{
			return m_render_idx;
		}
	};

	extern __constant__ ConstantKernel const_kernel[2];

	__host__ void copyConstantKernel(
		const ConstantKernel* hCudaConstantKernel,
		const uint32_t& update_idx,
		cudaStream_t& stream);
}

#endif