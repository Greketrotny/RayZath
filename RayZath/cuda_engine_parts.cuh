#ifndef CUDA_ENGINE_PARTS_CUH
#define CUDA_ENGINE_PARTS_CUH

#include "rzexception.h"
#include "camera.h"

#include <vector>

namespace RayZath
{
	namespace CudaEngine
	{
		struct HostPinnedMemory
		{
		private:
			void* mp_host_pinned_memory;
			uint32_t m_size;


		public:
			__host__ HostPinnedMemory() = delete;
			__host__ HostPinnedMemory(const HostPinnedMemory&) = delete;
			__host__ HostPinnedMemory(HostPinnedMemory&&) = delete;
			__host__ HostPinnedMemory(uint32_t m_size);
			__host__ ~HostPinnedMemory();


		public:
			__host__ void SetMemorySize(uint32_t bytes);
			__host__ void FreeMemory();
			__host__ void* GetPointerToMemory();
			__host__ uint32_t GetSize() const;
		};

		struct CudaDevice
		{
		private:
			uint32_t m_device_id;
			cudaDeviceProp m_device_prop;


		public:
			CudaDevice(const uint32_t& device_id);


		public:
			const uint32_t& GetDeviceId() const;
			const cudaDeviceProp& GetProperties() const;
		};
		struct CudaHardware
		{
		private:
			std::vector<CudaDevice> m_devices;


		public:
			CudaHardware();


		public:
			const CudaDevice& GetDevice(const uint32_t& id) const;
			uint32_t GetDeviceCount() const noexcept;
		};

		struct LaunchConfiguration
		{
		private:
			dim3 m_block;
			dim3 m_grid;
			uint32_t m_shared_mem_size;
			uint32_t m_device_id;
			uint32_t m_camera_id;
			const bool m_update;

		public:
			LaunchConfiguration(
				const CudaHardware& hardware,
				const Handle<Camera>& camera,
				bool update);


		public:
			dim3 GetThreadBlock() const noexcept;
			dim3 GetGrid() const noexcept;
			uint32_t GetSharedMemorySize() const noexcept;
			uint32_t GetDeviceId() const noexcept;
			uint32_t GetCameraId() const noexcept;
			bool GetUpdateFlag() const noexcept;
		};
	}
}

#endif