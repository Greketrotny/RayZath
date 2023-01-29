#include "cuda_engine_parts.cuh"
#include "cuda_kernel_data.cuh"
#include "cuda_render_parts.cuh"

#include "cuda_exception.hpp"
#include "cuda_include.hpp"

namespace RayZath::Cuda
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
			RZAssertCoreCUDA(cudaFreeHost(mp_host_pinned_memory));
		m_size = 0u;
	}

	void HostPinnedMemory::SetMemorySize(size_t bytes)
	{
		if (bytes == this->m_size) return;

		if (mp_host_pinned_memory)
			RZAssertCoreCUDA(cudaFreeHost(mp_host_pinned_memory));

		RZAssertCoreCUDA(cudaMallocHost((void**)&mp_host_pinned_memory, bytes));
		this->m_size = bytes;
	}
	void HostPinnedMemory::FreeMemory()
	{
		if (mp_host_pinned_memory)
			RZAssertCoreCUDA(cudaFreeHost(mp_host_pinned_memory));
		m_size = 0u;
	}
	void* HostPinnedMemory::GetPointerToMemory()
	{
		if (mp_host_pinned_memory == nullptr && m_size > 0u)
			RZAssertCoreCUDA(cudaMallocHost((void**)&mp_host_pinned_memory, m_size));

		return mp_host_pinned_memory;
	}
	size_t HostPinnedMemory::size() const
	{
		return m_size;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



	// ~~~~~~~~ [STRUCT] Device ~~~~~~~~
	Device::Device(uint32_t device_id)
		: m_device_id(device_id)
	{
		cudaGetDeviceProperties(&m_device_prop, m_device_id);
	}

	void Device::reset()
	{
		cudaSetDevice(m_device_id);
		cudaDeviceReset();
	}

	uint32_t Device::GetDeviceId() const
	{
		return m_device_id;
	}
	const cudaDeviceProp& Device::GetProperties() const
	{
		return m_device_prop;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	// ~~~~~~~~ [STRUCT] Hardware ~~~~~~~~
	Hardware::Hardware()
	{
		int count;
		RZAssertCoreCUDA(cudaGetDeviceCount(&count));
		for (int i = 0u; i < count; ++i)
		{
			m_devices.push_back(Device(uint32_t(i)));
		}
	}

	void Hardware::reset()
	{
		for (auto& d : m_devices)
		{
			d.reset();
		}
	}

	const Device& Hardware::GetDevice(uint32_t id) const
	{
		RZAssert(id < m_devices.size(), "Invalid device id");
		return m_devices[id];
	}
	uint32_t Hardware::GetDeviceCount() const noexcept
	{
		return uint32_t(m_devices.size());
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



	// ~~~~~~~~ [STRUCT] LaunchConfiguration ~~~~~~~~
	LaunchConfiguration::LaunchConfiguration(
		const Hardware& hardware,
		const Engine::Handle<Engine::Camera>& camera,
		bool update)
		: m_update(update)
	{
		RZAssert(
			hardware.GetDeviceCount() > 0u,
			"No cuda device available to construct launch configuration.");

		const Device& device = hardware.GetDevice(0);

		m_block = dim3(
			(unsigned int)(std::min(
				device.GetProperties().warpSize,
				device.GetProperties().maxThreadsDim[0])),
			(unsigned int)(std::min(
				device.GetProperties().maxThreadsPerBlock / device.GetProperties().warpSize / 2,
				device.GetProperties().maxThreadsDim[1])),
			1u);

		m_grid = dim3(
			std::min(
				(static_cast<uint32_t>(camera->width()) + m_block.x - 1u) / m_block.x,
				static_cast<uint32_t>(device.GetProperties().maxGridSize[0])),
			std::min(
				(static_cast<uint32_t>(camera->height()) + m_block.y - 1u) / m_block.y,
				static_cast<uint32_t>(device.GetProperties().maxGridSize[1])),
			1u);


		RZAssert(
			device.GetProperties().sharedMemPerBlock >= sizeof(Kernel::GlobalKernel),
			"not enough shared memory to hold GlobalKernel structure");
		m_shared_mem_size = sizeof(Kernel::GlobalKernel);

		RZAssert(
			device.GetProperties().totalConstMem >= sizeof(Kernel::ConstantKernel) * 2u /* 2u for double buffering */,
			"not enough constant memory to hold ConstantKernel structure");

		m_device_id = 0;
		m_camera_id = camera.accessor()->idx();
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


	// ~~~~~~~~ [STRUCT] launchConfigurations ~~~~~~~~
	void LaunchConfigurations::construct(
		const Hardware& hardware,
		const Engine::World& world,
		const bool update_flag)
	{
		m_configs.clear();
		for (uint32_t i = 0u; i < world.container<RayZath::Engine::ObjectType::Camera>().count(); ++i)
		{
			const auto& camera = world.container<RayZath::Engine::ObjectType::Camera>()[i];
			if (!camera) continue;
			if (!camera->enabled()) continue;

			m_configs.push_back(
				LaunchConfiguration(
					hardware, camera, camera->stateRegister().IsModified() || update_flag));
		}
	}
	const std::vector<LaunchConfiguration>& LaunchConfigurations::GetConfigs()
	{
		return m_configs;
	}
}