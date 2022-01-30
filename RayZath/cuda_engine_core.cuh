#ifndef CUDA_ENGINE_CORE_CUH
#define CUDA_ENGINE_CORE_CUH

#include "cuda_engine_renderer.cuh"

namespace RayZath::Cuda
{	
	class EngineCore
	{
	public:
		enum class State
		{
			None,
			Work,
			Wait
		};
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
		bool m_update_flag;

		HostPinnedMemory m_hpm_CudaWorld, m_hpm_CudaKernel;

		std::mutex m_mtx;

		State m_state;
		Stage m_stage;
		FenceTrack_t m_fence_track;

		TimeTable m_core_time_table, m_render_time_table;


	public:
		EngineCore();
		~EngineCore();


	public:
		void RenderWorld(
			RayZath::Engine::World& hWorld,
			const RayZath::Engine::RenderConfig& render_config,
			const bool block = true,
			const bool sync = true);
		void TransferResults();


	private:
		void CreateStreams();
		void DestroyStreams();
		void CreateGlobalKernels();
		void DestroyGlobalKernels();
		void ReconstructKernels();
		void CreateCudaWorld();
		void DestroyCudaWorld();
		void CopyCudaWorldDeviceToHost();
		void CopyCudaWorldHostToDevice();


	public:
		Hardware& GetHardware();
		Indexer& GetIndexer();
		Renderer& GetRenderer();
		LaunchConfigurations& GetLaunchConfigs(const bool idx);
		Kernel::GlobalKernel* GetGlobalKernel(const bool idx);
		World* GetCudaWorld();
		FenceTrack_t& GetFenceTrack();
		const TimeTable& GetCoreTimeTable() const;
		const TimeTable& GetRenderTimeTable() const;

		cudaStream_t& GetUpdateStream();
		cudaStream_t& GetRenderStream();

	public:
		const State& GetState();
		const Stage& GetStage();
	private:
		void SetState(const State& state);
		void SetStage(const Stage& stage);
	};
}

#endif // !CUDA_ENGINE_CORE_CUH
