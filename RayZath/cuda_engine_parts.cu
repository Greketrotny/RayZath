#include "cuda_engine_parts.cuh"
#include "cuda_render_parts.cuh"

namespace RayZath
{
	namespace CudaEngine
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
			m_size = 0u;
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

		void CudaDevice::Reset()
		{
			cudaSetDevice(m_device_id);
			cudaDeviceReset();
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
				m_devices.push_back(CudaDevice(uint32_t(i)));
			}
		}

		void CudaHardware::Reset()
		{
			for (auto& d : m_devices)
			{
				d.Reset();
			}
		}

		const CudaDevice& CudaHardware::GetDevice(const uint32_t& id) const
		{
			RZAssert(id < m_devices.size(), "Invalid device id");
			return m_devices[id];
		}
		uint32_t CudaHardware::GetDeviceCount() const noexcept
		{
			return uint32_t(m_devices.size());
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



		// ~~~~~~~~ [STRUCT] LaunchConfiguration ~~~~~~~~
		LaunchConfiguration::LaunchConfiguration(
			const CudaHardware& hardware,
			const Handle<Camera>& camera,
			bool update)
			: m_update(update)
		{
			RZAssert(
				hardware.GetDeviceCount() > 0u,
				"No cuda device available to construct launch configuration.");

			const CudaDevice& device = hardware.GetDevice(0);


			m_block = dim3(
				unsigned int(std::min(
					device.GetProperties().warpSize,
					device.GetProperties().maxThreadsDim[0])),
				unsigned int(std::min(
					//device.GetProperties().maxThreadsPerBlock / device.GetProperties().warpSize,
					16,
					device.GetProperties().maxThreadsDim[1])),
				1u);

			m_grid = dim3(
				std::min(
					(static_cast<uint32_t>(camera->GetWidth()) + m_block.x - 1u) / m_block.x,
					static_cast<uint32_t>(device.GetProperties().maxGridSize[0])),
				std::min(
					(static_cast<uint32_t>(camera->GetHeight()) + m_block.y - 1u) / m_block.y,
					static_cast<uint32_t>(device.GetProperties().maxGridSize[1])),
				1u);


			RZAssert(
				device.GetProperties().sharedMemPerBlock >= sizeof(CudaGlobalKernel),
				"not enough shared memory to hold CudaGlobalKernel structure");
			m_shared_mem_size = sizeof(CudaGlobalKernel);

			RZAssert(
				device.GetProperties().totalConstMem >= sizeof(CudaConstantKernel) * 2u /* 2u for double buffering */,
				"not enough constant memory to hold CudaConstantKernel structure");

			m_device_id = 0;
			m_camera_id = camera.GetResource()->GetId();
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
		void LaunchConfigurations::Construct(
			const CudaHardware& hardware,
			const World& world,
			const bool update_flag)
		{
			m_configs.clear();
			for (uint32_t i = 0u; i < world.Container<Camera>().GetCapacity(); ++i)
			{
				const Handle<Camera>& camera = world.Container<Camera>()[i];
				if (!camera) continue;	// no camera at the index
				if (!camera->Enabled()) continue;	// camera is disabled

				m_configs.push_back(
					LaunchConfiguration(
						hardware, camera, update_flag));
			}
		}
		const std::vector<LaunchConfiguration>& LaunchConfigurations::GetConfigs()
		{
			return m_configs;
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	}
}