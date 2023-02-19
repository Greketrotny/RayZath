#ifndef CUDA_OBJECT_CONTAINER_CUH
#define	CUDA_OBJECT_CONTAINER_CUH

#include "cuda_engine_parts.cuh"
#include "cuda_instance.cuh"

namespace RayZath::Cuda
{
	class World;

	template <class HostObject, class CudaObject>
	struct ObjectContainer
	{
	private:
		CudaObject* mp_storage = nullptr;
		uint32_t m_capacity = 0, m_count = 0;


	public:
		__host__ ~ObjectContainer()
		{
			if (m_count > 0u)
			{
				CudaObject* hostCudaObjects = (CudaObject*)malloc(m_count * sizeof(CudaObject));
				RZAssertCoreCUDA(cudaMemcpy(
					hostCudaObjects, mp_storage,
					m_count * sizeof(CudaObject),
					cudaMemcpyKind::cudaMemcpyDeviceToHost));

				// destroy all objects
				for (uint32_t i = 0u; i < m_count; ++i)
					hostCudaObjects[i].~CudaObject();

				free(hostCudaObjects);
			}

			RZAssertCoreCUDA(cudaFree(mp_storage));
		}


	public:
		__host__ void reconstruct(
			const World& hCudaWorld,
			RayZath::Engine::ObjectContainer<HostObject>& hContainer,
			HostPinnedMemory& hpm,
			cudaStream_t& mirror_stream)
		{
			if (!hContainer.stateRegister().IsModified()) return;

			const uint32_t hpm_chunk_size = uint32_t(hpm.size() / sizeof(CudaObject));
			RZAssertCore(hpm_chunk_size != 0u, "Not enough host pinned memory for reconstruction.");

			// allocate memory with new size, when current capacity doesn't match 
			// host capacity
			CudaObject* d_dst_storage = mp_storage;
			if (hContainer.capacity() != m_capacity)
				RZAssertCoreCUDA(cudaMalloc(&d_dst_storage, sizeof(CudaObject) * hContainer.capacity()));

			{
				/*
				* Perform reconstruction from source memory through host pinned memory
				* to destination memory
				*/

				const uint32_t end = std::min(hContainer.count(), m_count);
				for (uint32_t begin = 0u, count = hpm_chunk_size; begin < end; begin += count)
				{
					if (begin + count > end) count = end - begin;

					// copy to hCudaObjects memory from device
					CudaObject* hCudaObjects = (CudaObject*)hpm.GetPointerToMemory();
					RZAssertCoreCUDA(cudaMemcpyAsync(
						hCudaObjects, mp_storage + begin,
						count * sizeof(CudaObject),
						cudaMemcpyKind::cudaMemcpyDeviceToHost, mirror_stream));
					RZAssertCoreCUDA(cudaStreamSynchronize(mirror_stream));

					// loop through all objects in the current chunk of objects
					for (uint32_t i = 0u; i < count; ++i)
						hCudaObjects[i].reconstruct(hCudaWorld, hContainer[begin + i], mirror_stream);

					// copy mirrored objects back to device
					RZAssertCoreCUDA(cudaMemcpyAsync(
						d_dst_storage + begin, hCudaObjects,
						count * sizeof(CudaObject),
						cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
					RZAssertCoreCUDA(cudaStreamSynchronize(mirror_stream));
				}
			}

			{
				/*
				* Destroy every CudaObject not beeing a mirror of HostObject
				*/

				for (uint32_t begin = hContainer.count(), count = hpm_chunk_size;
					begin < m_count;
					begin += count)
				{
					if (begin + count > m_count) count = m_count - begin;

					// copy to hCudaObjects memory from device
					CudaObject* hCudaObjects = (CudaObject*)hpm.GetPointerToMemory();
					RZAssertCoreCUDA(cudaMemcpyAsync(
						hCudaObjects, mp_storage + begin,
						count * sizeof(CudaObject),
						cudaMemcpyKind::cudaMemcpyDeviceToHost, mirror_stream));
					RZAssertCoreCUDA(cudaStreamSynchronize(mirror_stream));

					// loop through all objects in the current chunk of objects
					for (uint32_t i = 0u; i < count; ++i)
						hCudaObjects[i].~CudaObject();

					// copy destroyed objects back to device
					RZAssertCoreCUDA(cudaMemcpyAsync(
						mp_storage + begin, hCudaObjects,
						count * sizeof(CudaObject),
						cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
					RZAssertCoreCUDA(cudaStreamSynchronize(mirror_stream));
				}
			}

			{
				/*
				* Construct new CudaObjects to reflect each new HostObject
				*/

				for (uint32_t begin = m_count, count = hpm_chunk_size;
					begin < hContainer.count();
					begin += count)
				{
					if (begin + count > hContainer.count())
						count = hContainer.count() - begin;

					// construct CudaObjects in host pinned memory
					CudaObject* hCudaObjects = (CudaObject*)hpm.GetPointerToMemory();
					for (uint32_t i = 0u; i < count; ++i)
					{
						new (&hCudaObjects[i]) CudaObject();
						hCudaObjects[i].reconstruct(hCudaWorld, hContainer[begin + i], mirror_stream);
					}

					// copy constructed objects back to device
					RZAssertCoreCUDA(cudaMemcpyAsync(
						d_dst_storage + begin, hCudaObjects,
						count * sizeof(CudaObject),
						cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
					RZAssertCoreCUDA(cudaStreamSynchronize(mirror_stream));
				}
			}

			if (mp_storage != d_dst_storage)
			{
				RZAssertCoreCUDA(cudaFree(mp_storage));
				mp_storage = d_dst_storage;
			}

			m_capacity = hContainer.capacity();
			m_count = hContainer.count();

			hContainer.stateRegister().MakeUnmodified();
		}
		__host__ void reconstruct(
			const World& hCudaWorld,
			RayZath::Engine::ObjectContainer<HostObject>& hContainer,
			const std::vector<uint32_t>& reordered_ids,
			HostPinnedMemory& hpm,
			cudaStream_t& mirror_stream)
		{
			if (!hContainer.stateRegister().IsModified()) return;

			const uint32_t hpm_chunk_size = uint32_t(hpm.size() / sizeof(CudaObject));
			RZAssertCore(hpm_chunk_size != 0u, "Not enough host pinned memory for reconstruction.");

			// allocate memory with new size, when current capacity doesn't match 
			// host capacity
			CudaObject* d_dst_storage = mp_storage;
			if (hContainer.capacity() != m_capacity)
				RZAssertCoreCUDA(cudaMalloc(&d_dst_storage, sizeof(CudaObject) * hContainer.capacity()));

			{
				/*
				* Perform reconstruction from source memory through host pinned memory
				* to destination memory
				*/

				const uint32_t end = std::min(hContainer.count(), m_count);
				for (uint32_t begin = 0u, count = hpm_chunk_size; begin < end; begin += count)
				{
					if (begin + count > end) count = end - begin;

					// copy to hCudaObjects memory from device
					CudaObject* hCudaObjects = (CudaObject*)hpm.GetPointerToMemory();
					RZAssertCoreCUDA(cudaMemcpyAsync(
						hCudaObjects, mp_storage + begin,
						count * sizeof(CudaObject),
						cudaMemcpyKind::cudaMemcpyDeviceToHost, mirror_stream));
					RZAssertCoreCUDA(cudaStreamSynchronize(mirror_stream));

					// loop through all objects in the current chunk of objects
					for (uint32_t i = 0u; i < count; ++i)
					{
						hCudaObjects[i].reconstruct(
							hCudaWorld,
							(begin + i < reordered_ids.size()) ? hContainer[reordered_ids[begin + i]] :
							RayZath::Engine::Handle<HostObject>{},
							mirror_stream);
					}

					// copy mirrored objects back to device
					RZAssertCoreCUDA(cudaMemcpyAsync(
						d_dst_storage + begin, hCudaObjects,
						count * sizeof(CudaObject),
						cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
					RZAssertCoreCUDA(cudaStreamSynchronize(mirror_stream));
				}
			}

			{
				/*
				* Destroy every CudaObject not beeing a mirror of HostObject
				*/

				for (uint32_t begin = hContainer.count(), count = hpm_chunk_size;
					begin < m_count;
					begin += count)
				{
					if (begin + count > m_count) count = m_count - begin;

					// copy to hCudaObjects memory from device
					CudaObject* hCudaObjects = (CudaObject*)hpm.GetPointerToMemory();
					RZAssertCoreCUDA(cudaMemcpyAsync(
						hCudaObjects, mp_storage + begin,
						count * sizeof(CudaObject),
						cudaMemcpyKind::cudaMemcpyDeviceToHost, mirror_stream));
					RZAssertCoreCUDA(cudaStreamSynchronize(mirror_stream));

					// loop through all objects in the current chunk of objects
					for (uint32_t i = 0u; i < count; ++i)
						hCudaObjects[i].~CudaObject();

					// copy destroyed objects back to device
					RZAssertCoreCUDA(cudaMemcpyAsync(
						mp_storage + begin, hCudaObjects,
						count * sizeof(CudaObject),
						cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
					RZAssertCoreCUDA(cudaStreamSynchronize(mirror_stream));
				}
			}

			{
				/*
				* Construct new CudaObjects to reflect each new HostObject
				*/

				for (uint32_t begin = m_count, count = hpm_chunk_size;
					begin < hContainer.count();
					begin += count)
				{
					if (begin + count > hContainer.count())
						count = hContainer.count() - begin;

					// construct CudaObjects in host pinned memory
					CudaObject* hCudaObjects = (CudaObject*)hpm.GetPointerToMemory();
					for (uint32_t i = 0u; i < count; ++i)
					{
						new (&hCudaObjects[i]) CudaObject();
						hCudaObjects[i].reconstruct(
							hCudaWorld,
							(begin + i < reordered_ids.size())
							? hContainer[reordered_ids[begin + i]]
							: RayZath::Engine::Handle<HostObject>{},
							mirror_stream);
					}

					// copy constructed objects back to device
					RZAssertCoreCUDA(cudaMemcpyAsync(
						d_dst_storage + begin, hCudaObjects,
						count * sizeof(CudaObject),
						cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
					RZAssertCoreCUDA(cudaStreamSynchronize(mirror_stream));
				}
			}

			if (mp_storage != d_dst_storage)
			{
				RZAssertCoreCUDA(cudaFree(mp_storage));
				mp_storage = d_dst_storage;
			}

			m_capacity = hContainer.capacity();
			m_count = hContainer.count();

			hContainer.stateRegister().MakeUnmodified();
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
		__host__ const CudaObject* storageAddress() const
		{
			return mp_storage;
		}
		__host__ CudaObject* storageAddress()
		{
			return mp_storage;
		}
	private:
		__host__ __device__ __inline__ const uint32_t& capacity() const
		{
			return m_capacity;
		}
	public:
		__host__ __device__ __inline__ const uint32_t& count() const
		{
			return m_count;
		}
		__host__ __device__ bool empty() const
		{
			return m_count == 0u;
		}
	};
}

#endif // !CUDA_OBJECT_CONTAINER_CUH