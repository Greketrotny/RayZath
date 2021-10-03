#include "cuda_render_parts.cuh"
#include "rzexception.h"
#include "cuda_world.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		// ~~~~~~~~ [STRUCT] RNG ~~~~~~~~
		void RNG::Reconstruct()
		{
			// generate random numbers
			for (uint32_t i = 0u; i < s_count; ++i)
				m_unsigned_uniform[i] = (rand() % RAND_MAX) / static_cast<float>(RAND_MAX);
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		// ~~~~~~~~ [SRUCT] Seeds ~~~~~~~~
		void Seeds::Reconstruct(cudaStream_t& stream)
		{
			// generate random seeds
			for (uint32_t i = 0u; i < s_count; ++i)
				m_seeds[i] = rand() % s_count;
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



		// ~~~~~~~~ [STRUCT] CudaConstantKernel ~~~~~~~~
		void CudaConstantKernel::Reconstruct()
		{
			m_rng.Reconstruct();
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		// ~~~~~~~~ [CLASS] CudaGlobalKernel ~~~~~~~~
		CudaGlobalKernel::CudaGlobalKernel()
			: m_render_idx(0u)
		{}
		CudaGlobalKernel::~CudaGlobalKernel()
		{}

		void CudaGlobalKernel::Reconstruct(
			uint32_t render_idx,
			cudaStream_t& stream)
		{
			m_render_idx = render_idx;
			m_seeds.Reconstruct(stream);
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


		// ~~~~~~~~ [STRUCT] CudaTriangle ~~~~~~~~
		CudaTriangle::CudaTriangle(const Triangle& hostTriangle)
		{
			normal = hostTriangle.normal;
			material_id = hostTriangle.material_id & 0x3F;
		}

		void CudaTriangle::SetVertices(const vec3f& v1, const vec3f& v2, const vec3f& v3)
		{
			this->v1 = v1;
			this->v2 = v2;
			this->v3 = v3;
		}
		void CudaTriangle::SetTexcrds(const vec2f& t1, const vec2f& t2, const vec2f& t3)
		{
			this->t1 = t1;
			this->t2 = t2;
			this->t3 = t3;
		}
		void CudaTriangle::SetNormals(const vec3f& n1, const vec3f& n2, const vec3f& n3)
		{
			this->n1 = n1;
			this->n2 = n2;
			this->n3 = n3;
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	}
}