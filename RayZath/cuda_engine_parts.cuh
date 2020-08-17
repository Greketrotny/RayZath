#ifndef CUDA_ENGINE_PARTS_CUH
#define CUDA_ENGINE_PARTS_CUH

#include "rzexception.h"
#include "camera.h"

#include <vector>

namespace RayZath
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
		size_t m_device_id;
		dim3 m_thread_block;


	public:
		CudaDevice(const size_t& device_id, const cudaDeviceProp& device_prop);


	public:
		dim3 GetThreadBlock() const;
	};
	struct CudaHardware
	{
	private:
		std::vector<CudaDevice> m_devices;


	public:
		CudaHardware();


	public:
		const CudaDevice& GetDevice(const size_t& id) const;
		size_t GetDeviceCount() const noexcept;
	};

	struct LaunchConfiguration
	{
	private:
		dim3 m_grid;
		dim3 m_block;
		size_t m_shared_mem_size;
		size_t m_device_id;
		size_t m_camera_id;
		const bool m_update;

	public:
		LaunchConfiguration(
			const CudaHardware& hardware,
			const Camera& camera,
			bool update);


	public:
		dim3 GetGrid() const noexcept;
		dim3 GetThreadBlock() const noexcept;
		size_t GetSharedMemorySize() const noexcept;
		size_t GetDeviceId() const noexcept;
		size_t GetCameraId() const noexcept;
		bool GetUpdateFlag() const noexcept;
	};
}

#endif