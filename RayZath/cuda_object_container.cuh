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
				if (m_count > 0u)
				{
					// allocate host memory
					CudaObject* hostCudaObjects = (CudaObject*)malloc(m_count * sizeof(CudaObject));
					CudaErrorCheck(cudaMemcpy(
						hostCudaObjects, mp_storage,
						m_count * sizeof(CudaObject),
						cudaMemcpyKind::cudaMemcpyDeviceToHost));

					// destroy all objects
					for (uint32_t i = 0u; i < m_count; ++i)
						hostCudaObjects[i].~CudaObject();

					// free host memory
					free(hostCudaObjects);
				}

				// free storage
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

				const uint32_t hpm_chunk_size = uint32_t(hpm.GetSize() / sizeof(CudaObject));
				if (hpm_chunk_size == 0u)
					ThrowException("Not enough host pinned memory for reconstruction.");

				// allocate memory with new size, when current capacity doesn't match 
				// host capacity
				CudaObject* d_dst_storage = mp_storage;
				if (hContainer.GetCapacity() != m_capacity)
					CudaErrorCheck(cudaMalloc(&d_dst_storage, sizeof(CudaObject) * hContainer.GetCapacity()));

				{
					/*
					* Perform reconstruction from source memory through host pinned memory
					* to destination memory
					*/

					const uint32_t end = std::min(hContainer.GetCount(), m_count);
					for (uint32_t begin = 0u, count = hpm_chunk_size; begin < end; begin += count)
					{
						if (begin + count > end) count = end - begin;

						// copy to hCudaObjects memory from device
						CudaObject* hCudaObjects = (CudaObject*)hpm.GetPointerToMemory();
						CudaErrorCheck(cudaMemcpyAsync(
							hCudaObjects, mp_storage + begin,
							count * sizeof(CudaObject),
							cudaMemcpyKind::cudaMemcpyDeviceToHost, mirror_stream));
						CudaErrorCheck(cudaStreamSynchronize(mirror_stream));

						// loop through all objects in the current chunk of objects
						for (uint32_t i = 0u; i < count; ++i)
							hCudaObjects[i].Reconstruct(hCudaWorld, hContainer[begin + i], mirror_stream);

						// copy mirrored objects back to device
						CudaErrorCheck(cudaMemcpyAsync(
							d_dst_storage + begin, hCudaObjects,
							count * sizeof(CudaObject),
							cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
						CudaErrorCheck(cudaStreamSynchronize(mirror_stream));
					}
				}

				{
					/*
					* Destroy every CudaObject not beeing a mirror of HostObject
					*/

					for (uint32_t begin = hContainer.GetCount(), count = hpm_chunk_size;
						begin < m_count;
						begin += count)
					{
						if (begin + count > m_count) count = m_count - begin;

						// copy to hCudaObjects memory from device
						CudaObject* hCudaObjects = (CudaObject*)hpm.GetPointerToMemory();
						CudaErrorCheck(cudaMemcpyAsync(
							hCudaObjects, mp_storage + begin,
							count * sizeof(CudaObject),
							cudaMemcpyKind::cudaMemcpyDeviceToHost, mirror_stream));
						CudaErrorCheck(cudaStreamSynchronize(mirror_stream));

						// loop through all objects in the current chunk of objects
						for (uint32_t i = 0u; i < count; ++i)
							hCudaObjects[i].~CudaObject();

						// copy destroyed objects back to device
						CudaErrorCheck(cudaMemcpyAsync(
							mp_storage + begin, hCudaObjects,
							count * sizeof(CudaObject),
							cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
						CudaErrorCheck(cudaStreamSynchronize(mirror_stream));
					}
				}

				{
					/*
					* Construct new CudaObjects to reflect each new HostObject
					*/

					for (uint32_t begin = m_count, count = hpm_chunk_size;
						begin < hContainer.GetCount();
						begin += count)
					{
						if (begin + count > hContainer.GetCount())
							count = hContainer.GetCount() - begin;

						// construct CudaObjects in host pinned memory
						CudaObject* hCudaObjects = (CudaObject*)hpm.GetPointerToMemory();
						for (uint32_t i = 0u; i < count; ++i)
						{
							new (&hCudaObjects[i]) CudaObject();
							hCudaObjects[i].Reconstruct(hCudaWorld, hContainer[begin + i], mirror_stream);
						}

						// copy constructed objects back to device
						CudaErrorCheck(cudaMemcpyAsync(
							d_dst_storage + begin, hCudaObjects,
							count * sizeof(CudaObject),
							cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
						CudaErrorCheck(cudaStreamSynchronize(mirror_stream));
					}
				}

				if (mp_storage != d_dst_storage)
				{
					CudaErrorCheck(cudaFree(mp_storage));
					mp_storage = d_dst_storage;
				}

				m_capacity = hContainer.GetCapacity();
				m_count = hContainer.GetCount();

				hContainer.GetStateRegister().MakeUnmodified();
			}
			__host__ void Reconstruct(
				const CudaWorld& hCudaWorld,
				ObjectContainer<HostObject>& hContainer,
				const std::vector<uint32_t>& reordered_ids,
				HostPinnedMemory& hpm,
				cudaStream_t& mirror_stream)
			{
				if (!hContainer.GetStateRegister().IsModified()) return;

				const uint32_t hpm_chunk_size = uint32_t(hpm.GetSize() / sizeof(CudaObject));
				if (hpm_chunk_size == 0u)
					ThrowException("Not enough host pinned memory for reconstruction.");

				// allocate memory with new size, when current capacity doesn't match 
				// host capacity
				CudaObject* d_dst_storage = mp_storage;
				if (hContainer.GetCapacity() != m_capacity)
					CudaErrorCheck(cudaMalloc(&d_dst_storage, sizeof(CudaObject) * hContainer.GetCapacity()));

				{
					/*
					* Perform reconstruction from source memory through host pinned memory
					* to destination memory
					*/

					const uint32_t end = std::min(hContainer.GetCount(), m_count);
					for (uint32_t begin = 0u, count = hpm_chunk_size; begin < end; begin += count)
					{
						if (begin + count > end) count = end - begin;

						// copy to hCudaObjects memory from device
						CudaObject* hCudaObjects = (CudaObject*)hpm.GetPointerToMemory();
						CudaErrorCheck(cudaMemcpyAsync(
							hCudaObjects, mp_storage + begin,
							count * sizeof(CudaObject),
							cudaMemcpyKind::cudaMemcpyDeviceToHost, mirror_stream));
						CudaErrorCheck(cudaStreamSynchronize(mirror_stream));

						// loop through all objects in the current chunk of objects
						for (uint32_t i = 0u; i < count; ++i)
							hCudaObjects[i].Reconstruct(
								hCudaWorld,
								hContainer[reordered_ids[begin + i]],
								mirror_stream);

						// copy mirrored objects back to device
						CudaErrorCheck(cudaMemcpyAsync(
							d_dst_storage + begin, hCudaObjects,
							count * sizeof(CudaObject),
							cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
						CudaErrorCheck(cudaStreamSynchronize(mirror_stream));
					}
				}

				{
					/*
					* Destroy every CudaObject not beeing a mirror of HostObject
					*/

					for (uint32_t begin = hContainer.GetCount(), count = hpm_chunk_size;
						begin < m_count;
						begin += count)
					{
						if (begin + count > m_count) count = m_count - begin;

						// copy to hCudaObjects memory from device
						CudaObject* hCudaObjects = (CudaObject*)hpm.GetPointerToMemory();
						CudaErrorCheck(cudaMemcpyAsync(
							hCudaObjects, mp_storage + begin,
							count * sizeof(CudaObject),
							cudaMemcpyKind::cudaMemcpyDeviceToHost, mirror_stream));
						CudaErrorCheck(cudaStreamSynchronize(mirror_stream));

						// loop through all objects in the current chunk of objects
						for (uint32_t i = 0u; i < count; ++i)
							hCudaObjects[i].~CudaObject();

						// copy destroyed objects back to device
						CudaErrorCheck(cudaMemcpyAsync(
							mp_storage + begin, hCudaObjects,
							count * sizeof(CudaObject),
							cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
						CudaErrorCheck(cudaStreamSynchronize(mirror_stream));
					}
				}

				{
					/*
					* Construct new CudaObjects to reflect each new HostObject
					*/

					for (uint32_t begin = m_count, count = hpm_chunk_size;
						begin < hContainer.GetCount();
						begin += count)
					{
						if (begin + count > hContainer.GetCount())
							count = hContainer.GetCount() - begin;

						// construct CudaObjects in host pinned memory
						CudaObject* hCudaObjects = (CudaObject*)hpm.GetPointerToMemory();
						for (uint32_t i = 0u; i < count; ++i)
						{
							new (&hCudaObjects[i]) CudaObject();
							hCudaObjects[i].Reconstruct(
								hCudaWorld, 
								hContainer[reordered_ids[begin + i]], 
								mirror_stream);
						}

						// copy constructed objects back to device
						CudaErrorCheck(cudaMemcpyAsync(
							d_dst_storage + begin, hCudaObjects,
							count * sizeof(CudaObject),
							cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
						CudaErrorCheck(cudaStreamSynchronize(mirror_stream));
					}
				}

				if (mp_storage != d_dst_storage)
				{
					CudaErrorCheck(cudaFree(mp_storage));
					mp_storage = d_dst_storage;
				}

				m_capacity = hContainer.GetCapacity();
				m_count = hContainer.GetCount();

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
		private:
			__host__ __device__ __inline__ const uint32_t& GetCapacity() const
			{
				return m_capacity;
			}
		public:
			__host__ __device__ __inline__ const uint32_t& GetCount() const
			{
				return m_count;
			}
		};
	}
}

#endif // !CUDA_OBJECT_CONTAINER_CUH