#include "cuda_render_parts.cuh"
#include "rzexception.h"
#include "cuda_world.cuh"

#include <random>

namespace RayZath::Cuda
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


	// ~~~~~~~~ [STRUCT] Triangle ~~~~~~~~
	Triangle::Triangle(const RayZath::Engine::Triangle& hostTriangle)
	{
		m_normal = hostTriangle.normal;
		m_material_id = hostTriangle.material_id & 0x3F;
	}

	void Triangle::SetVertices(const vec3f& v1, const vec3f& v2, const vec3f& v3)
	{
		m_v1 = v1;
		m_v2 = v2;
		m_v3 = v3;
	}
	void Triangle::SetTexcrds(const vec2f& t1, const vec2f& t2, const vec2f& t3)
	{
		m_t1 = t1;
		m_t2 = t2;
		m_t3 = t3;
	}
	void Triangle::SetNormals(const vec3f& n1, const vec3f& n2, const vec3f& n3)
	{
		m_n1 = n1;
		m_n2 = n2;
		m_n3 = n3;
	}
}