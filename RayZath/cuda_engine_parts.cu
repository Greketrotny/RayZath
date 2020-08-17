#include "cuda_engine_parts.cuh"

namespace RayZath
{
	// ~~~~~~~~ [STRUCT] HostPinnedMemory ~~~~~~~~
	HostPinnedMemory::HostPinnedMemory(size_t m_size)
		: mp_host_pinned_memory(nullptr)
	{
		this->m_size = m_size;
	}
	HostPinnedMemory::~HostPinnedMemory()
	{
		if (mp_host_pinned_memory)
			CudaErrorCheck(cudaFreeHost(mp_host_pinned_memory));
		m_size = size_t(0u);
	}

	void HostPinnedMemory::SetMemorySize(size_t bytes)
	{
		if (bytes == this->m_size) return;

		if (mp_host_pinned_memory)
			CudaErrorCheck(cudaFreeHost(mp_host_pinned_memory));

		CudaErrorCheck(cudaMallocHost((void**)&mp_host_pinned_memory, bytes));
		this->m_size = bytes;
	}
	void HostPinnedMemory::FreeMemory()
	{
		if (mp_host_pinned_memory)
			CudaErrorCheck(cudaFreeHost(mp_host_pinned_memory));
		m_size = size_t(0u);
	}
	void* HostPinnedMemory::GetPointerToMemory()
	{
		if (mp_host_pinned_memory == nullptr && m_size > 0u)
			CudaErrorCheck(cudaMallocHost((void**)&mp_host_pinned_memory, m_size));

		return mp_host_pinned_memory;
	}
	size_t HostPinnedMemory::GetSize() const
	{
		return m_size;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



	// ~~~~~~~~ [STRUCT] CudaDevice ~~~~~~~~
	CudaDevice::CudaDevice(const size_t& device_id, const cudaDeviceProp& device_prop)
		: m_device_id(device_id)
	{
		m_thread_block = dim3(device_prop.maxThreadsPerBlock, 1u, 1u);
	}

	dim3 CudaDevice::GetThreadBlock() const
	{
		return m_thread_block;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	// ~~~~~~~~ [STRUCT] CudaHardware ~~~~~~~~
	CudaHardware::CudaHardware()
	{
		int count;
		cudaGetDeviceCount(&count);
		for (int i = 0u; i < count; ++i)
		{
			cudaSetDevice(i);

			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, i);
			m_devices.push_back(CudaDevice(i, deviceProp));
		}
	}

	const CudaDevice& CudaHardware::GetDevice(const size_t& id) const
	{
		ThrowAtCondition(id < m_devices.size(), L"Invalid device id");
		return m_devices[id];
	}
	size_t CudaHardware::GetDeviceCount() const noexcept
	{
		return m_devices.size();
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



	// ~~~~~~~~ [STRUCT] LaunchConfiguration ~~~~~~~~
	LaunchConfiguration::LaunchConfiguration(
		const CudaHardware& hardware,
		const Camera& camera,
		bool update)
		: m_update(update)
	{
		const CudaDevice& device = hardware.GetDevice(0);

		m_grid = dim3(
			(camera.GetWidth() * camera.GetHeight() + device.GetThreadBlock().x) /
			device.GetThreadBlock().x, 1u, 1u);
		m_block = device.GetThreadBlock();
		m_shared_mem_size = 0u;

		m_device_id = 0;
		m_camera_id = camera.GetId();
	}

	dim3 LaunchConfiguration::GetGrid() const noexcept
	{
		return m_grid;
	}
	dim3 LaunchConfiguration::GetThreadBlock() const noexcept
	{
		return m_block;
	}
	size_t LaunchConfiguration::GetSharedMemorySize() const noexcept
	{
		return m_shared_mem_size;
	}
	size_t LaunchConfiguration::GetDeviceId() const noexcept
	{
		return m_device_id;
	}
	size_t LaunchConfiguration::GetCameraId() const noexcept
	{
		return m_camera_id;
	}
	bool LaunchConfiguration::GetUpdateFlag() const noexcept
	{
		return m_update;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}