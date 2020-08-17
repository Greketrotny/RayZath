#ifndef CUDA_ENGINE_H
#define CUDA_ENGINE_H

#include "cuda_world.cuh"
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

		CudaWorld* mp_dCudaWorld;
		HostPinnedMemory m_hpm_CudaWorld;

		cudaStream_t m_mirror_stream, m_render_stream;

		bool m_update_flag;
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
		struct Timer
		{
			std::chrono::time_point<std::chrono::high_resolution_clock> start;

			Timer()
			{
				Start();
			}

			void Start()
			{
				start = std::chrono::high_resolution_clock::now();
			}

			float PeekElapsedTime()
			{
				auto stop = std::chrono::high_resolution_clock::now();
				std::chrono::duration<float, std::milli> duration = stop - start;
				return duration.count();
			}
			float GetElapsedTime()
			{
				auto stop = std::chrono::high_resolution_clock::now();
				std::chrono::duration<float, std::milli> duration = stop - start;
				start = stop;
				return duration.count();
			}
		};
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
		void ReconstructCudaWorld(
			CudaWorld* dCudaWorld,
			World& hWorld,
			cudaStream_t* mirror_stream);
		void TransferResultsToHost(
			CudaWorld* dCudaWorld, 
			World& hWorld, 
			cudaStream_t* mirror_stream);

		void LaunchFunction();
	};
}

#endif // CUDA_ENGINE_CORE_H