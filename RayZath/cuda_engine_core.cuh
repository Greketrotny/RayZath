#ifndef CUDA_ENGINE_CORE_CUH
#define CUDA_ENGINE_CORE_CUH

#include "cuda_engine_parts.cuh"
#include "cuda_world.cuh"

#include <stdint.h>
#include <thread>

namespace RayZath
{
	namespace CudaEngine
	{
		class CudaEngineCore;
		struct CudaRenderer;

		struct CudaIndexer
		{
		private:
			bool m_update_idx, m_render_idx;

		public:
			CudaIndexer();

		public:
			const bool& UpdateIdx() const;
			const bool& RenderIdx() const;
			void Swap();
		};

		template <size_t GC>
		struct FenceTrack
		{
		private:
			std::array<ThreadGate, GC> m_gates;


		public:
			FenceTrack(const bool all_opened)
			{
				if (all_opened)
				{
					for (auto& g : m_gates)
					{
						g.Open();
					}
				}
				else
				{
					for (auto& g : m_gates)
					{
						g.Close();
					}
				}
			}
			~FenceTrack()
			{
				for (auto& g : m_gates)
				{
					g.Open();
				}
			}


		public:
			void WaitForEndOfAndClose(const size_t idx)
			{
				m_gates[idx].WaitAndClose();
			}
			void WaitForEndOf(const size_t idx)
			{
				m_gates[idx].Wait();
			}
			void CloseGate(const size_t& idx)
			{
				m_gates[idx].Close();
			}
			void OpenGate(const size_t& idx)
			{
				m_gates[idx].Open();
			}
			void OpenAll()
			{
				for (auto& g : m_gates)
				{
					g.Open();
				}
			}
		};

		struct TimeTable
		{
		private:
			Timer m_timer, m_cycle_timer;
			std::vector<std::pair<std::string, float>> m_stamps;

		public:
			TimeTable();

		public:
			void AppendStage(const std::string& s);
			void AppendFullCycle(const std::string& s);
			void ResetTable();
			void ResetTime();
			std::string ToString(const uint32_t width) const;
		};
		struct CudaRenderer
		{
			enum class State
			{
				None,
				Idle,
				Work,
				Wait
			};
			enum class Stage
			{
				None,
				Idle,
				Preprocess,
				MainRender,
				Postprocess
			};
		private:
			CudaEngineCore* const mp_engine_core;

			std::unique_ptr<std::thread> mp_render_thread;
			std::atomic<bool> m_is_thread_alive, m_terminate_thread;
			ThreadGate* mp_blocking_gate;

			std::unique_ptr<Exception> m_exception;
			std::unique_ptr<CudaException> m_cuda_exception;

			State m_state;
			Stage m_stage;
			FenceTrack<5> m_fence_track;

			TimeTable m_time_table;

			std::mutex m_mtx;


		public:
			CudaRenderer(CudaEngineCore* const engine_core);
			~CudaRenderer();


		public:
			void LaunchThread();
			void TerminateThread();


			FenceTrack<5>& GetFenceTrack();
			const TimeTable& GetTimeTable() const;
			const State& GetState() const;
			const Stage& GetStage() const;

		private:
			void SetState(const State& state);
			void SetStage(const Stage& stage);

		private:
			void RenderFunctionWrapper();
			void RenderFunction() noexcept;
			bool CheckTermination();

			void ReportException(const Exception& e);
			void ReportCudaException(const CudaException& e);
			void ResetExceptions();
		public:
			void ThrowIfException();
		};

		class CudaEngineCore
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
				ResultTransfer
			};
		private:
			CudaHardware m_hardware;
			CudaIndexer m_indexer;
			CudaRenderer m_renderer;
			LaunchConfigurations m_configs[2];
			CudaGlobalKernel* mp_global_kernel[2];
			CudaWorld* mp_dCudaWorld, *mp_hCudaWorld;
			World* mp_hWorld;
			cudaStream_t m_update_stream, m_render_stream;
			bool m_update_flag;

			HostPinnedMemory m_hpm_CudaWorld, m_hpm_CudaKernel;

			std::mutex m_mtx;			

			State m_state;
			Stage m_stage;
			FenceTrack<5> m_fence_track;

			TimeTable m_core_time_table, m_render_time_table;


		public:
			CudaEngineCore();
			~CudaEngineCore();


		public:
			void RenderWorld(
				World& hWorld, 
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
			CudaHardware& GetHardware();
			CudaIndexer& GetIndexer();
			CudaRenderer& GetRenderer();
			LaunchConfigurations& GetLaunchConfigs(const bool idx);
			CudaGlobalKernel* GetGlobalKernel(const bool idx);
			CudaWorld* GetCudaWorld();
			FenceTrack<5>& GetFenceTrack();
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
}

#endif // !CUDA_ENGINE_CORE_CUH
