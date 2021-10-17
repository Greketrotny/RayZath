#include "cuda_render_parts.cuh"
#include "rzexception.h"
#include "cuda_world.cuh"

#include <random>

namespace RayZath
{
	namespace CudaEngine
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



		// ~~~~~~~~ [STRUCT] CudaConstantKernel ~~~~~~~~
		void CudaConstantKernel::Reconstruct()
		{
			m_seeds.Reconstruct();
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		// ~~~~~~~~ [CLASS] CudaGlobalKernel ~~~~~~~~
		CudaGlobalKernel::CudaGlobalKernel()
			: m_render_idx(0u)
		{}

		void CudaGlobalKernel::Reconstruct(
			uint32_t render_idx,
			cudaStream_t& stream)
		{
			m_render_idx = render_idx;
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


		// ~~~~~~~~ [STRUCT] CudaTriangle ~~~~~~~~
		CudaTriangle::CudaTriangle(const Triangle& hostTriangle)
		{
			m_normal = hostTriangle.normal;
			m_material_id = hostTriangle.material_id & 0x3F;
		}

		void CudaTriangle::SetVertices(const vec3f& v1, const vec3f& v2, const vec3f& v3)
		{
			m_v1 = v1;
			m_v2 = v2;
			m_v3 = v3;
		}
		void CudaTriangle::SetTexcrds(const vec2f& t1, const vec2f& t2, const vec2f& t3)
		{
			m_t1 = t1;
			m_t2 = t2;
			m_t3 = t3;
		}
		void CudaTriangle::SetNormals(const vec3f& n1, const vec3f& n2, const vec3f& n3)
		{
			m_n1 = n1;
			m_n2 = n2;
			m_n3 = n3;
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	}
}