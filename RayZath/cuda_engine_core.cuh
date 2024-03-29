#ifndef CUDA_ENGINE_CORE_CUH
#define CUDA_ENGINE_CORE_CUH

#include "cuda_engine_renderer.cuh"

namespace RayZath::Cuda
{	
	class EngineCore
	{
	public:
		enum class Stage
		{
			None,
			AsyncReconstruction,
			WorldReconstruction,
			CameraReconstruction,
			Synchronization,
			ResultTransfer
		};
	private:
		typedef FenceTrack<6> FenceTrack_t;
		Hardware m_hardware;
		Indexer m_indexer;
		Renderer m_renderer;
		LaunchConfigurations m_configs[2];
		Kernel::GlobalKernel* mp_global_kernel[2];
		RayZath::Engine::RenderConfig m_render_config;
		World* mp_dCudaWorld, * mp_hCudaWorld;
		RayZath::Engine::World* mp_hWorld;
		cudaStream_t m_update_stream, m_render_stream;

		HostPinnedMemory m_hpm_CudaWorld, m_hpm_CudaKernel;

		std::mutex m_mtx;

		FenceTrack_t m_fence_track;

		RayZath::Engine::TimeTable m_core_time_table, m_render_time_table;


	public:
		EngineCore();
		~EngineCore();


	public:
		void renderWorld(
			RayZath::Engine::World& hWorld,
			const RayZath::Engine::RenderConfig& render_config,
			const bool block = true,
			const bool sync = true);
		void CopyRenderToHost();


	private:
		void createStreams();
		void destroyStreams();
		void createGlobalKernels();
		void destroyGlobalKernels();
		void reconstructKernels();
		void createCudaWorld();
		void destroyCudaWorld();
		void copyCudaWorldDeviceToHost();
		void copyCudaWorldHostToDevice();


	public:
		Hardware& hardware();
		Indexer& indexer();
		Renderer& renderer();
		LaunchConfigurations& launchConfigs(const bool idx);
		Kernel::GlobalKernel* globalKernel(const bool idx);
		const RayZath::Engine::RenderConfig& renderConfig() const;
		World* cudaWorld();
		FenceTrack_t& fenceTrack();
		const RayZath::Engine::TimeTable& coreTimeTable() const;
		const RayZath::Engine::TimeTable& renderTimeTable() const;

		cudaStream_t& updateStream();
		cudaStream_t& renderStream();
	};
}

#endif // !CUDA_ENGINE_CORE_CUH
