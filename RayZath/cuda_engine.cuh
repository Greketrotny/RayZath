#ifndef CUDA_ENGINE_H
#define CUDA_ENGINE_H

#include "cuda_world.cuh"
#include "engine_parts.h"
#include "cuda_render_parts.cuh"
#include "cuda_engine_parts.cuh"

#include "cuda_render_kernel.cuh"
#include "cuda_postprocess_kernel.cuh"
#include "rzexception.h"

#include <thread>
#include <condition_variable>
#include <mutex>
#include <chrono>

#include <string>
#include <sstream>
#include <iomanip>
#include <vector>

namespace RayZath
{
	namespace CudaEngine
	{
		class Engine
		{
		private:
			CudaHardware m_hardware;

			uint32_t m_update_ix = 0u, m_render_ix = 1u;
			std::vector<LaunchConfiguration> m_launch_configs[2];
			CudaGlobalKernel* mp_global_kernel[2];

			CudaWorld* mp_dCudaWorld;
			HostPinnedMemory m_hpm_CudaWorld;
			HostPinnedMemory m_hpm_CudaKernel;

			cudaStream_t m_mirror_stream, m_render_stream;

			bool m_update_flag;

			std::thread* mp_launch_thread = nullptr;
			bool m_launch_thread_terminate = false;
			ThreadGate m_host_gate, m_kernel_gate;
		public:

			// debug staff -------------------------------- //
			struct DebugInfo
			{
			private:
				std::vector<std::wstring> debugStrings;
			public:
				void AddDebugString(std::wstring newDebugString)
				{
					debugStrings.push_back(newDebugString);
				}
				void Clear()
				{
					debugStrings.clear();
				}
				std::wstring InfoToString()
				{
					std::wstring fullDebugString = L"";
					for (const auto& s : debugStrings)
					{
						fullDebugString += s + L"\n";
					}
					return fullDebugString;
				}
			};
			DebugInfo mainDebugInfo;
			std::wstring renderTimingString;

			void AppendTimeToString(std::wstring& str, const std::wstring& measurement, const float& value)
			{
				std::wstringstream ss;
				ss.fill(L' ');
				ss.width(30);
				ss << measurement;

				ss.width();
				ss.precision(3);
				ss << std::fixed << value << "ms\n";

				str += ss.str();
			}
			// debug staff -------------------------------- //


		public:
			Engine();
			~Engine();


		public:
			void RenderWorld(World& hWorld);
		private:
			void CreateLaunchConfigurations(const World& world);
			void ReconstructKernelData(cudaStream_t& mirror_stream);
			void ReconstructCudaWorld(
				CudaWorld* dCudaWorld,
				World& hWorld,
				cudaStream_t& mirror_stream);
			void TransferResultsToHost(
				CudaWorld* dCudaWorld,
				World& hWorld,
				cudaStream_t& mirror_stream);

			void LaunchFunction();
		};
	}
}

#endif // CUDA_ENGINE_CORE_H