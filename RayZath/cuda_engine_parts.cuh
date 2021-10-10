#ifndef CUDA_ENGINE_PARTS_CUH
#define CUDA_ENGINE_PARTS_CUH

#include "rzexception.h"
#include "camera.h"
#include "engine_parts.h"

#include "world.h"

#include <vector>
#include <array>

namespace RayZath
{
	namespace CudaEngine
	{
		struct HostPinnedMemory
		{
		private:
			void* mp_host_pinned_memory;
			size_t m_size;


		public:
			__host__ HostPinnedMemory() = delete;
			__host__ HostPinnedMemory(const HostPinnedMemory&) = delete;
			__host__ HostPinnedMemory(HostPinnedMemory&&) = delete;
			__host__ HostPinnedMemory(size_t m_size);
			__host__ ~HostPinnedMemory();


		public:
			__host__ void SetMemorySize(size_t bytes);
			__host__ void FreeMemory();
			__host__ void* GetPointerToMemory();
			__host__ size_t GetSize() const;
		};

		struct CudaDevice
		{
		private:
			uint32_t m_device_id;
			cudaDeviceProp m_device_prop;


		public:
			CudaDevice(uint32_t device_id);


		public:
			void Reset();

			uint32_t GetDeviceId() const;
			const cudaDeviceProp& GetProperties() const;
		};
		struct CudaHardware
		{
		private:
			std::vector<CudaDevice> m_devices;


		public:
			CudaHardware();


		public:
			void Reset();

			const CudaDevice& GetDevice(uint32_t id) const;
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
		struct LaunchConfigurations
		{
		private:
			std::vector<LaunchConfiguration> m_configs;


		public:
			void Construct(
				const CudaHardware& hardware,
				const World& world,
				const bool update_flag);

			const std::vector<LaunchConfiguration>& GetConfigs();
		};
	}
}

#endif