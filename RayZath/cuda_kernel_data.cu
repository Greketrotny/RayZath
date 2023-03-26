#include "cuda_kernel_data.cuh"

#include "engine_parts.hpp"
#include "cuda_exception.hpp"

#include <random>
#include <algorithm>

namespace RayZath::Cuda::Kernel
{
	// ~~~~~~~~ [SRUCT] Seeds ~~~~~~~~
	void Seeds::reconstruct()
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<float> dis(-10.0f, 10.0f);

		for (uint32_t i = 0u; i < s_count; ++i)
			m_seeds[i] = dis(gen);
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	// ~~~~~~~~ RenderConfig::LightSampling ~~~~~~~~
	__host__ RenderConfig::LightSampling& RenderConfig::LightSampling::operator=(
		const RayZath::Engine::LightSampling& light_sampling)
	{
		// at least one to avoid division by zero
		m_spot_light = std::max(light_sampling.spotLight(), uint8_t(1));
		m_direct_light = std::max(light_sampling.directLight(), uint8_t(1));

		return *this;
	}
	// ~~~~~~~~ RenderConfig::Tracing ~~~~~~~~
	__host__ RenderConfig::Tracing& RenderConfig::Tracing::operator=(
		const RayZath::Engine::Tracing& tracing)
	{
		m_max_depth = tracing.maxDepth();

		return *this;
	}

	// ~~~~~~~~ RenderConfig ~~~~~~~~
	RenderConfig& RenderConfig::operator=(const RayZath::Engine::RenderConfig& render_config)
	{
		m_light_sampling = render_config.lightSampling();
		m_tracing = render_config.tracing();

		return *this;
	}


	// ~~~~~~~~ [STRUCT] ConstantKernel ~~~~~~~~
	void ConstantKernel::reconstruct(const RayZath::Engine::RenderConfig& render_config)
	{
		m_seeds.reconstruct();
		m_render_config = render_config;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	// ~~~~~~~~ [CLASS] GlobalKernel ~~~~~~~~
	GlobalKernel::GlobalKernel()
		: m_render_idx(0u)
	{}

	void GlobalKernel::reconstruct(
		uint32_t render_idx,
		cudaStream_t& stream)
	{
		m_render_idx = render_idx;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	__constant__ ConstantKernel const_kernel[2];

	__host__ void copyConstantKernel(
		const ConstantKernel* hCudaConstantKernel,
		const uint32_t& update_idx,
		cudaStream_t& stream)
	{
		RZAssertCoreCUDA(cudaMemcpyToSymbolAsync(
			(const void*)const_kernel, hCudaConstantKernel,
			sizeof(ConstantKernel), update_idx * sizeof(ConstantKernel),
			cudaMemcpyKind::cudaMemcpyHostToDevice, stream));
		RZAssertCoreCUDA(cudaStreamSynchronize(stream));
	}
}