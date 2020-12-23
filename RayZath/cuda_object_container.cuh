#ifndef CUDA_OBJECT_CONTAINER_CUH
#define	CUDA_OBJECT_CONTAINER_CUH

#include "cuda_engine_parts.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		class CudaWorld;

		template <class HostObject, class CudaObject>
		struct CudaObjectContainer
		{
		private:
			CudaObject* mp_storage;
			uint32_t m_capacity, m_count;


		public:
			__host__ CudaObjectContainer()
				: mp_storage(nullptr)
				, m_capacity(0u)
				, m_count(0u)
			{}
			__host__ ~CudaObjectContainer()
			{
				if (this->m_capacity > 0u)
				{
					// allocate host memory
					CudaObject* hostCudaObjects = (CudaObject*)malloc(m_capacity * sizeof(CudaObject));
					CudaErrorCheck(cudaMemcpy(
						hostCudaObjects, mp_storage,
						m_capacity * sizeof(CudaObject),
						cudaMemcpyKind::cudaMemcpyDeviceToHost));

					// destroy all objects
					for (uint32_t i = 0u; i < m_capacity; ++i)
					{
						hostCudaObjects[i].~CudaObject();
					}

					// free host memory
					free(hostCudaObjects);
				}

				// free objects' arrays
				CudaErrorCheck(cudaFree(mp_storage));
				mp_storage = nullptr;

				m_capacity = 0u;
				m_count = 0u;
			}


		public:
			__host__ void Reconstruct(
				const CudaWorld& hCudaWorld,
				ObjectContainer<HostObject>& hContainer,
				HostPinnedMemory& hpm,
				cudaStream_t& mirror_stream)
			{
				if (!hContainer.GetStateRegister().IsModified()) return;

				if (hContainer.GetCapacity() != m_capacity)
				{// storage sizes don't match

					// allocate memory
					CudaObject* hCudaObjects = (CudaObject*)malloc(
						std::max(m_capacity, hContainer.GetCapacity()) * sizeof(CudaObject));

					if (m_capacity > 0u)
					{
						// copy object data to host
						CudaErrorCheck(cudaMemcpy(
							hCudaObjects, mp_storage,
							m_capacity * sizeof(CudaObject),
							cudaMemcpyKind::cudaMemcpyDeviceToHost));

						// destroy each object
						for (uint32_t i = 0u; i < m_capacity; ++i)
						{
							hCudaObjects[i].~CudaObject();
						}

						// free objects array
						if (mp_storage) CudaErrorCheck(cudaFree(mp_storage));
						m_capacity = 0u;
						m_count = 0u;
					}

					if (hContainer.GetCapacity() > 0u)
					{
						m_capacity = hContainer.GetCapacity();
						m_count = hContainer.GetCount();

						// allocate new amounts of memory for objects
						CudaErrorCheck(cudaMalloc(&mp_storage, m_capacity * sizeof(CudaObject)));

						// reconstruct all objects
						for (uint32_t i = 0u; i < hContainer.GetCapacity(); ++i)
						{
							new (&hCudaObjects[i]) CudaObject();

							if (hContainer[i])	hCudaObjects[i].Reconstruct(hCudaWorld, hContainer[i], mirror_stream);
							else				hCudaObjects[i].MakeNotExist();
						}

						// copy object to device
						CudaErrorCheck(cudaMemcpy(
							mp_storage, hCudaObjects,
							m_capacity * sizeof(CudaObject),
							cudaMemcpyKind::cudaMemcpyHostToDevice));
					}

					free(hCudaObjects);
				}
				else
				{// Asynchronous mirroring

					// divide work into chunks of objects to fit in page-locked memory
					uint32_t chunkSize = hpm.GetSize() / sizeof(CudaObject);
					if (chunkSize == 0) return;		// TODO: throw exception

					for (uint32_t startIndex = 0u; startIndex < m_capacity; startIndex += chunkSize)
					{
						if (startIndex + chunkSize > m_capacity) chunkSize = m_capacity - startIndex;

						// copy to hCudaObjects memory from device
						CudaObject* hCudaObjects = (CudaObject*)hpm.GetPointerToMemory();
						CudaErrorCheck(cudaMemcpyAsync(
							hCudaObjects, mp_storage + startIndex,
							chunkSize * sizeof(CudaObject),
							cudaMemcpyKind::cudaMemcpyDeviceToHost, mirror_stream));
						CudaErrorCheck(cudaStreamSynchronize(mirror_stream));

						// loop through all objects in the current chunk of objects
						for (uint32_t i = 0u; i < chunkSize; ++i)
						{
							if (hContainer[startIndex + i])
								hCudaObjects[i].Reconstruct(hCudaWorld, hContainer[startIndex + i], mirror_stream);
							else
								hCudaObjects[i].MakeNotExist();
						}

						// copy mirrored objects back to device
						CudaErrorCheck(cudaMemcpyAsync(
							mp_storage + startIndex, hCudaObjects,
							chunkSize * sizeof(CudaObject),
							cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
						CudaErrorCheck(cudaStreamSynchronize(mirror_stream));
					}
				}

				hContainer.GetStateRegister().MakeUnmodified();
			}


		public:
			__host__ __device__ __inline__ CudaObject& operator[](uint32_t index)
			{
				return mp_storage[index];
			}
			__host__ __device__ __inline__ const CudaObject& operator[](uint32_t index) const
			{
				return mp_storage[index];
			}


		public:
			__host__ const CudaObject* GetStorageAddress() const
			{
				return mp_storage;
			}
			__host__ CudaObject* GetStorageAddress()
			{
				return mp_storage;
			}
			__host__ __device__ __inline__ const uint32_t& GetCapacity() const
			{
				return m_capacity;
			}
			__host__ __device__ __inline__ const uint32_t& GetCount() const
			{
				return m_count;
			}
		};
	}
}

#endif // !CUDA_OBJECT_CONTAINER_CUH