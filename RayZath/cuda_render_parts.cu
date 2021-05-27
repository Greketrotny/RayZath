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
			: v1(nullptr), v2(nullptr), v3(nullptr)
			, t1(nullptr), t2(nullptr), t3(nullptr)
			, n1(nullptr), n2(nullptr), n3(nullptr)
		{
			this->normal = hostTriangle.normal;
			this->material_id = hostTriangle.material_id & 0x3F;
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	}
}