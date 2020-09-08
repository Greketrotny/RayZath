#ifndef CUDA_BVH_H
#define CUDA_BVH_H

#include "bvh.h"
#include "cuda_include.h"

namespace RayZath
{
	template <class HB, class CudaObject>
	class CudaBVH
	{
	private:



	public:
		CudaBVH();
		~CudaBVH();


	public:
		void Reconstruct(
			BVH<HB>& hBVH, 
			cudaStream_t& mirror_stream);
	};
}

#endif // !CUDA_BVH_H