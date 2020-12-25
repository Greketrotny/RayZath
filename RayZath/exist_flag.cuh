#ifndef WITH_EXIST_FLAG_H
#define WITH_EXIST_FLAG_H

#include "cuda_include.h"

namespace RayZath
{
	namespace CudaEngine
	{
		class WithExistFlag
		{
		private:
			bool exist;


		public:
			__host__ __device__ WithExistFlag()
				:exist(true)
			{}
			__host__ __device__ ~WithExistFlag()
			{
				exist = false;
			}


		public:
			__host__ void MakeNotExist()
			{
				exist = false;
			}
			__host__ __device__ __inline__ const bool& Exist() const
			{
				return exist;
			}
		};
	}
}

#endif // !WITH_EXIST_FLAG_H