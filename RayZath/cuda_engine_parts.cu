#include "cuda_engine_parts.cuh"
#include "cuda_render_parts.cuh"

namespace RayZath
{
	// ~~~~~~~~ [STRUCT] HostPinnedMemory ~~~~~~~~
	HostPinnedMemory::HostPinnedMemory(uint32_t m_size)
		: mp_host_pinned_memory(nullptr)
	{
		this->m_size = m_size;
	}
	HostPinnedMemory::~HostPinnedMemory()
	{
		if (mp_host_pinned_memory)
			CudaErrorCheck(cudaFreeHost(mp_host_pinned_memory));
		m_size = uint32_t(0u);
	}

	void HostPinnedMemory::SetMemorySize(uint32_t bytes)
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
		m_size = uint32_t(0u);
	}
	void* HostPinnedMemory::GetPointerToMemory()
	{
		if (mp_host_pinned_memory == nullptr && m_size > 0u)
			CudaErrorCheck(cudaMallocHost((void**)&mp_host_pinned_memory, m_size));

		return mp_host_pinned_memory;
	}
	uint32_t HostPinnedMemory::GetSize() const
	{
		return m_size;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



	// ~~~~~~~~ [STRUCT] CudaDevice ~~~~~~~~
	CudaDevice::CudaDevice(const uint32_t& device_id)
		: m_device_id(device_id)
	{
		cudaGetDeviceProperties(&m_device_prop, m_device_id);
	}

	const uint32_t& CudaDevice::GetDeviceId() const
	{
		return m_device_id;
	}
	const cudaDeviceProp& CudaDevice::GetProperties() const
	{
		return m_device_prop;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	// ~~~~~~~~ [STRUCT] CudaHardware ~~~~~~~~
	CudaHardware::CudaHardware()
	{
		int count;
		CudaErrorCheck(cudaGetDeviceCount(&count));
		for (int i = 0u; i < count; ++i)
		{
			m_devices.push_back(CudaDevice(i));
		}
	}

	const CudaDevice& CudaHardware::GetDevice(const uint32_t& id) const
	{
		ThrowAtCondition(id < m_devices.size(), L"Invalid device id");
		return m_devices[id];
	}
	uint32_t CudaHardware::GetDeviceCount() const noexcept
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
		ThrowAtCondition(
			hardware.GetDeviceCount() > 0u, 
			L"No cuda device available to construct launch configuration.");

		const CudaDevice& device = hardware.GetDevice(0);


		m_block = dim3(
			std::min(
				device.GetProperties().warpSize, 
				device.GetProperties().maxThreadsDim[0]),
			std::min(
				//device.GetProperties().maxThreadsPerBlock / device.GetProperties().warpSize,
				16,
				device.GetProperties().maxThreadsDim[1]),
			1u);

		m_grid = dim3(
			std::min(
				(static_cast<unsigned int>(camera.GetWidth()) + m_block.x - 1u) / m_block.x,
				static_cast<unsigned int>(device.GetProperties().maxGridSize[0])),
			std::min(
				(static_cast<unsigned int>(camera.GetHeight()) + m_block.y - 1u) / m_block.y,
				static_cast<unsigned int>(device.GetProperties().maxGridSize[1])),
			1u);


		//ThrowAtCondition(device.GetProperties().sharedMemPerBlock <= sizeof(CudaKernelData), L"shared memory");
		m_shared_mem_size = sizeof(CudaKernelData);

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
	uint32_t LaunchConfiguration::GetSharedMemorySize() const noexcept
	{
		return m_shared_mem_size;
	}
	uint32_t LaunchConfiguration::GetDeviceId() const noexcept
	{
		return m_device_id;
	}
	uint32_t LaunchConfiguration::GetCameraId() const noexcept
	{
		return m_camera_id;
	}
	bool LaunchConfiguration::GetUpdateFlag() const noexcept
	{
		return m_update;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}