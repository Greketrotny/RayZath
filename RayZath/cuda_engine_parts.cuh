#ifndef CUDA_ENGINE_PARTS_CUH
#define CUDA_ENGINE_PARTS_CUH

#include "rzexception.hpp"
#include "roho.hpp"

#include <vector>
#include <array>

namespace RayZath::Engine
{
	class World;
	class Camera;
}

namespace RayZath::Cuda
{
	struct HostPinnedMemory
	{
	private:
		void* mp_host_pinned_memory;
		std::size_t m_size;


	public:
		__host__ HostPinnedMemory() = delete;
		__host__ HostPinnedMemory(const HostPinnedMemory&) = delete;
		__host__ HostPinnedMemory(HostPinnedMemory&&) = delete;
		__host__ HostPinnedMemory(std::size_t m_size);
		__host__ ~HostPinnedMemory();


	public:
		__host__ void SetMemorySize(std::size_t bytes);
		__host__ void FreeMemory();
		__host__ void* GetPointerToMemory();
		__host__ std::size_t size() const;
	};

	struct Device
	{
	private:
		uint32_t m_device_id;
		cudaDeviceProp m_device_prop;


	public:
		Device(uint32_t device_id);


	public:
		void reset();

		uint32_t GetDeviceId() const;
		const cudaDeviceProp& GetProperties() const;
	};
	struct Hardware
	{
	private:
		std::vector<Device> m_devices;


	public:
		Hardware();


	public:
		void reset();

		const Device& GetDevice(uint32_t id) const;
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
			const Hardware& hardware,
			const RayZath::Engine::Handle<RayZath::Engine::Camera>& camera,
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
		void construct(
			const Hardware& hardware,
			const RayZath::Engine::World& world,
			const bool update_flag);

		const std::vector<LaunchConfiguration>& GetConfigs();
	};
}

#endif