#ifndef CUDA_ENGINE_H
#define CUDA_ENGINE_H

#include "cuda_world.cuh"
#include "engine_parts.h"
#include "cuda_render_parts.cuh"
#include "cuda_engine_parts.cuh"

#include "cuda_engine_kernel.cuh"
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
	class CudaEngine
	{
	private:
		CudaHardware m_hardware;

		size_t m_render_ix = 1, m_update_ix = 0;
		std::vector<LaunchConfiguration> m_launch_configs[2];
		CudaKernelData* mp_kernel_data[2];

		CudaWorld* mp_dCudaWorld;
		HostPinnedMemory m_hpm_CudaWorld;
		HostPinnedMemory m_hpm_CudaKernelData;

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
				for (auto s : debugStrings)
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
		CudaEngine();
		~CudaEngine();


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

#endif // CUDA_ENGINE_CORE_H