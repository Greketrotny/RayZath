#include "cuda_kernel_data.cuh"

#include <random>

namespace RayZath::Cuda::Kernel
{
	// ~~~~~~~~ [SRUCT] Seeds ~~~~~~~~
	void Seeds::Reconstruct()
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
		m_spot_light = light_sampling.GetSpotLight();
		m_direct_light = light_sampling.GetDirectLight();

		return *this;
	}

	// ~~~~~~~~ RenderConfig ~~~~~~~~
	RenderConfig& RenderConfig::operator=(const RayZath::Engine::RenderConfig& render_config)
	{
		m_light_sampling = render_config.GetLightSampling();

		return *this;
	}


	// ~~~~~~~~ [STRUCT] ConstantKernel ~~~~~~~~
	void ConstantKernel::Reconstruct(const RayZath::Engine::RenderConfig& render_config)
	{
		m_seeds.Reconstruct();
		m_render_config = render_config;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	// ~~~~~~~~ [CLASS] GlobalKernel ~~~~~~~~
	GlobalKernel::GlobalKernel()
		: m_render_idx(0u)
	{}

	void GlobalKernel::Reconstruct(
		uint32_t render_idx,
		cudaStream_t& stream)
	{
		m_render_idx = render_idx;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	__constant__ ConstantKernel const_kernel[2];

	__host__ void CopyConstantKernel(
		const ConstantKernel* hCudaConstantKernel,
		const uint32_t& update_idx,
		cudaStream_t& stream)
	{
		CudaErrorCheck(cudaMemcpyToSymbolAsync(
			(const void*)const_kernel, hCudaConstantKernel,
			sizeof(ConstantKernel), update_idx * sizeof(ConstantKernel),
			cudaMemcpyKind::cudaMemcpyHostToDevice, stream));
		CudaErrorCheck(cudaStreamSynchronize(stream));
	}
}