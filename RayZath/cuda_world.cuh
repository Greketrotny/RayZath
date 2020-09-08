#ifndef CUDA_WORLD_H
#define CUDA_WORLD_H

#include "world.h"
#include "cuda_engine_parts.cuh"

#include "cuda_camera.cuh"

#include "cuda_point_light.cuh"
#include "cuda_spot_light.cuh"
#include "cuda_direct_light.cuh"

#include "cuda_mesh.cuh"
#include "cuda_sphere.cuh"

namespace RayZath
{
	template <class HC, class CudaObject> struct CudaObjectContainer
	{
	private:
		CudaObject* mp_storage;
		uint64_t m_capacity, m_count;


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
				for (uint64_t i = 0u; i < m_capacity; ++i)
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
			HC& hostObjectContainer,
			HostPinnedMemory& hostPinnedMemory,
			cudaStream_t& mirrorStream)
		{
			if (hostObjectContainer.GetCapacity() != m_capacity)
			{// storage sizes don't match

				// allocate memory
				CudaObject* hostCudaObjects = (CudaObject*)malloc(
					std::max(m_capacity, hostObjectContainer.GetCapacity()) * sizeof(CudaObject));

				if (m_capacity > 0u)
				{
					// copy object data to host
					CudaErrorCheck(cudaMemcpy(
						hostCudaObjects, mp_storage,
						m_capacity * sizeof(CudaObject),
						cudaMemcpyKind::cudaMemcpyDeviceToHost));

					// destroy each object
					for (uint64_t i = 0u; i < m_capacity; ++i)
					{
						hostCudaObjects[i].~CudaObject();
					}

					// free objects array
					if (mp_storage) CudaErrorCheck(cudaFree(mp_storage));
					m_capacity = 0u;
					m_count = 0u;
				}

				if (hostObjectContainer.GetCapacity() > 0u)
				{
					m_capacity = hostObjectContainer.GetCapacity();
					m_count = hostObjectContainer.GetCount();

					// allocate new amounts of memory for objects
					CudaErrorCheck(cudaMalloc(&mp_storage, m_capacity * sizeof(CudaObject)));

					// reconstruct all objects
					for (uint64_t i = 0u; i < hostObjectContainer.GetCapacity(); ++i)
					{
						new (&hostCudaObjects[i]) CudaObject();

						if (hostObjectContainer[i]) hostCudaObjects[i].Reconstruct(*hostObjectContainer[i], mirrorStream);
						else						hostCudaObjects[i].MakeNotExist();
					}

					// copy object to device
					CudaErrorCheck(cudaMemcpy(
						mp_storage, hostCudaObjects,
						m_capacity * sizeof(CudaObject),
						cudaMemcpyKind::cudaMemcpyHostToDevice));
				}

				free(hostCudaObjects);
			}
			else
			{// Asynchronous mirroring

				// divide work into chunks of objects to fit in page-locked memory
				size_t chunkSize = hostPinnedMemory.GetSize() / sizeof(CudaObject);
				if (chunkSize == 0) return;		// TODO: throw exception

				for (size_t startIndex = 0, endIndex; startIndex < m_capacity; startIndex += chunkSize)
				{
					if (startIndex + chunkSize > m_capacity) chunkSize = m_capacity - startIndex;
					endIndex = startIndex + chunkSize;

					// copy to hostCudaObjects memory from device
					CudaObject* hostCudaObjects = (CudaObject*)hostPinnedMemory.GetPointerToMemory();
					CudaErrorCheck(cudaMemcpyAsync(
						hostCudaObjects, mp_storage + startIndex,
						chunkSize * sizeof(CudaObject),
						cudaMemcpyKind::cudaMemcpyDeviceToHost, mirrorStream));
					CudaErrorCheck(cudaStreamSynchronize(mirrorStream));

					// loop through all objects in the current chunk of objects
					for (size_t i = startIndex, j = 0; i < endIndex; ++i, ++j)
					{
						if (hostObjectContainer[i]) hostCudaObjects[i].Reconstruct(*hostObjectContainer[i], mirrorStream);
						else						hostCudaObjects[i].MakeNotExist();
					}

					// copy mirrored objects back to device
					CudaErrorCheck(cudaMemcpyAsync(
						mp_storage + startIndex, hostCudaObjects,
						chunkSize * sizeof(CudaObject),
						cudaMemcpyKind::cudaMemcpyHostToDevice, mirrorStream));
					CudaErrorCheck(cudaStreamSynchronize(mirrorStream));
				}
			}

			hostObjectContainer.Updated();
		}


	public:
		__host__ __device__ __inline__ CudaObject& operator[](size_t index)
		{
			return mp_storage[index];
		}
		__host__ __device__ __inline__ const CudaObject& operator[](size_t index) const
		{
			return mp_storage[index];
		}


	public:
		__host__ __device__ __inline__ const uint64_t& GetCapacity() const
		{
			return m_capacity;
		}
		__host__ __device__ __inline__ const uint64_t& GetCount() const
		{
			return m_count;
		}
	};

	class CudaWorld
	{
	public:
		CudaObjectContainer<ObjectContainer<Camera>, CudaCamera> cameras;
		CudaObjectContainer<ObjectContainer<PointLight>, CudaPointLight> pointLights;
		CudaObjectContainer<ObjectContainer<SpotLight>, CudaSpotLight> spotLights;
		CudaObjectContainer<ObjectContainer<DirectLight>, CudaDirectLight> directLights;
		CudaObjectContainer<ObjectContainer<Mesh>, CudaMesh> meshes;
		CudaObjectContainer<ObjectContainer<Sphere>, CudaSphere> spheres;


	public:
		static HostPinnedMemory m_hpm;


	public:
		__host__ CudaWorld();
		__host__ CudaWorld(const CudaWorld&) = delete;
		__host__ CudaWorld(CudaWorld&&) = delete;
		__host__ ~CudaWorld();


	public:
		__host__ void Reconstruct(
			World& host_world,
			cudaStream_t& mirror_stream);
	};
}

#endif // !CUDA_WORLD_H