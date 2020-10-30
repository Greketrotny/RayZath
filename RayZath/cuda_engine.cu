#include "cuda_engine.cuh"
#include "point.h"

namespace RayZath
{
	namespace CudaEngine
	{
		Engine::Engine()
			: mp_dCudaWorld(nullptr)
			, m_hpm_CudaWorld(sizeof(CudaWorld))
			, m_hpm_CudaKernelData(sizeof(CudaKernelData))
			, m_update_flag(true)
		{
			cudaSetDevice(0);

			// create streams
			CudaErrorCheck(cudaStreamCreate(&m_mirror_stream));
			CudaErrorCheck(cudaStreamCreate(&m_render_stream));

			// create empty CudaKernelData
			CudaKernelData* hCudaKernelData =
				(CudaKernelData*)m_hpm_CudaKernelData.GetPointerToMemory();
			for (uint32_t i = 0; i < sizeof(mp_kernel_data) / sizeof(*mp_kernel_data); ++i)
			{
				//new (hCudaKernelData) CudaKernelData();
				hCudaKernelData->renderIndex = 0u;

				CudaErrorCheck(cudaMalloc(
					(void**)&mp_kernel_data[i], sizeof(CudaKernelData)));
				CudaErrorCheck(cudaMemcpy(mp_kernel_data[i], hCudaKernelData,
					sizeof(CudaKernelData), cudaMemcpyKind::cudaMemcpyHostToDevice));
			}


			// create empty dCudaWorld
			CudaWorld* hCudaWorld = (CudaWorld*)m_hpm_CudaWorld.GetPointerToMemory();
			new (hCudaWorld) CudaWorld();
			CudaErrorCheck(cudaMalloc(&mp_dCudaWorld, sizeof(CudaWorld)));
			CudaErrorCheck(cudaMemcpy(
				mp_dCudaWorld, hCudaWorld,
				sizeof(CudaWorld),
				cudaMemcpyKind::cudaMemcpyHostToDevice));

			// create and launch kernel launching thread
			mp_launch_thread = new std::thread(&Engine::LaunchFunction, this);
			mp_launch_thread->detach();
		}
		Engine::~Engine()
		{
			// terminate and delete kernel launching thread
			m_launch_thread_terminate = true;
			m_kernel_gate.Open();
			m_host_gate.WaitForOpen();
			delete mp_launch_thread;

			// destroy mp_dCudaWorld
			if (mp_dCudaWorld)
			{
				CudaWorld* hCudaWorld = (CudaWorld*)m_hpm_CudaWorld.GetPointerToMemory();
				CudaErrorCheck(cudaMemcpy(
					hCudaWorld, mp_dCudaWorld,
					sizeof(CudaWorld),
					cudaMemcpyKind::cudaMemcpyDeviceToHost));

				hCudaWorld->~CudaWorld();

				CudaErrorCheck(cudaFree(mp_dCudaWorld));
				mp_dCudaWorld = nullptr;
			}

			// destroy dCudaKernelData
			CudaKernelData* hCudaKernelData =
				(CudaKernelData*)m_hpm_CudaKernelData.GetPointerToMemory();
			for (uint32_t i = 0; i < sizeof(mp_kernel_data) / sizeof(*mp_kernel_data); ++i)
			{
				CudaErrorCheck(cudaMemcpy(
					hCudaKernelData, mp_kernel_data[i],
					sizeof(CudaKernelData),
					cudaMemcpyKind::cudaMemcpyDeviceToHost));

				hCudaKernelData->~CudaKernelData();

				CudaErrorCheck(cudaFree(mp_kernel_data[i]));
				mp_kernel_data[i] = nullptr;
			}

			// destroy streams
			CudaErrorCheck(cudaStreamDestroy(m_mirror_stream));
			CudaErrorCheck(cudaStreamDestroy(m_render_stream));
		}

		void Engine::RenderWorld(World& hWorld)
		{
			mainDebugInfo.Clear();

			m_update_flag = hWorld.GetStateRegister().IsModified();
			Timer function_timer, step_timer;
			std::wstring timing_string = L"Host side:\n";


			// [>] Create Launch configurations
			step_timer.Start();
			CreateLaunchConfigurations(hWorld);
			AppendTimeToString(timing_string, L"create launch configs: ", step_timer.GetTime());


			// [>] Reconstruct CudaKernelData
			step_timer.Start();
			ReconstructKernelData(m_mirror_stream);
			AppendTimeToString(timing_string, L"reconstruct kernel data: ", step_timer.GetTime());


			// [>] Synchronize with kernel function
			step_timer.Start();
			m_host_gate.WaitForOpen();	// wait for kernel to finish render
			m_host_gate.Close();		// close gate for itself
			AppendTimeToString(timing_string, L"wait for kernel: ", step_timer.GetTime());
			mainDebugInfo.AddDebugString(renderTimingString);


			// [>] Reconstruct dCudaWorld
			step_timer.Start();
			hWorld.Update();
			ReconstructCudaWorld(mp_dCudaWorld, hWorld, m_mirror_stream);
			AppendTimeToString(timing_string, L"reconstruct CudaWorld: ", step_timer.GetTime());

			CudaKernel::CopyToConstantMemory(mp_kernel_data[m_update_ix], m_mirror_stream);

			// [>] Swap indexes
			std::swap(m_update_ix, m_render_ix);


			// [>] Launch kernel
			m_kernel_gate.Open();	// open gate for kernel
			//m_host_gate.WaitForOpen(); // <- uncoment for sync rendering


			// [>] Transfer results to host
			step_timer.Start();
			TransferResultsToHost(mp_dCudaWorld, hWorld, m_mirror_stream);
			AppendTimeToString(timing_string, L"copy final render to host: ", step_timer.GetTime());


			// [>] Sum up timings and add debug string
			AppendTimeToString(timing_string, L"render function full time: ", function_timer.GetTime());
			mainDebugInfo.AddDebugString(timing_string);

			std::wstring camera_str = L"camera: ";
			camera_str += std::to_wstring(hWorld.GetCameras()[0]->GetSamplesCount()) + L"spp\n";
			mainDebugInfo.AddDebugString(camera_str);
		}
		void Engine::CreateLaunchConfigurations(const World& world)
		{
			m_launch_configs[m_update_ix].clear();
			for (uint32_t i = 0; i < world.GetCameras().GetCapacity(); ++i)
			{
				const Camera* camera = world.GetCameras()[i];
				if (camera == nullptr) continue;	// no camera at the index
				if (!camera->Enabled()) continue;	// camera is disabled

				m_launch_configs[m_update_ix].push_back(
					LaunchConfiguration(
						m_hardware, *camera, m_update_flag));
			}
		}
		void Engine::ReconstructKernelData(cudaStream_t& mirror_stream)
		{
			CudaKernelData* hCudaKernelData =
				(CudaKernelData*)m_hpm_CudaKernelData.GetPointerToMemory();

			// copy dCudaKernelData to host
			CudaErrorCheck(cudaMemcpyAsync(
				hCudaKernelData,
				mp_kernel_data[m_update_ix],
				sizeof(CudaKernelData),
				cudaMemcpyKind::cudaMemcpyDeviceToHost,
				mirror_stream));
			CudaErrorCheck(cudaStreamSynchronize(mirror_stream));

			// reconstruct hCudaKernelData
			hCudaKernelData->Reconstruct(
				m_update_ix,
				mirror_stream);

			// copy hCudaKernelData to device
			CudaErrorCheck(cudaMemcpyAsync(
				mp_kernel_data[m_update_ix],
				hCudaKernelData,
				sizeof(CudaKernelData),
				cudaMemcpyKind::cudaMemcpyHostToDevice,
				mirror_stream));
			CudaErrorCheck(cudaStreamSynchronize(mirror_stream));
		}
		void Engine::ReconstructCudaWorld(
			CudaWorld* dCudaWorld,
			World& hWorld,
			cudaStream_t& mirror_stream)
		{
			if (!hWorld.GetStateRegister().IsModified()) return;

			// copy CudaWorld to host
			CudaWorld* hCudaWorld = (CudaWorld*)m_hpm_CudaWorld.GetPointerToMemory();
			CudaErrorCheck(cudaMemcpyAsync(
				hCudaWorld, dCudaWorld,
				sizeof(CudaWorld),
				cudaMemcpyKind::cudaMemcpyDeviceToHost, mirror_stream));
			CudaErrorCheck(cudaStreamSynchronize(mirror_stream));

			// reconstruct CudaWorld on host
			hCudaWorld->Reconstruct(hWorld, mirror_stream);

			// copy CudaWorld back to device
			CudaErrorCheck(cudaMemcpyAsync(
				dCudaWorld, hCudaWorld,
				sizeof(CudaWorld),
				cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
			CudaErrorCheck(cudaStreamSynchronize(mirror_stream));

			hWorld.GetStateRegister().MakeUnmodified();
		}
		void Engine::TransferResultsToHost(
			CudaWorld* dCudaWorld,
			World& hWorld,
			cudaStream_t& mirror_stream)
		{
			for (uint32_t i = 0; i < hWorld.GetCameras().GetCapacity(); ++i)
			{
				// check if hostCamera does exict
				Camera* hCamera = hWorld.GetCameras()[i];
				if (hCamera == nullptr) continue;	// no camera at this address
				if (!hCamera->Enabled()) continue;	// camera is disabled


				// [>] Get CudaWorld from device
				CudaWorld* hCudaWorld = (CudaWorld*)m_hpm_CudaWorld.GetPointerToMemory();
				CudaErrorCheck(cudaMemcpyAsync(
					hCudaWorld, dCudaWorld,
					sizeof(CudaWorld),
					cudaMemcpyKind::cudaMemcpyDeviceToHost, mirror_stream));
				CudaErrorCheck(cudaStreamSynchronize(mirror_stream));

				if (hCudaWorld->cameras.GetCount() == 0) return;	// hCudaWorld has no cameras


				// [>] Get CudaCamera class from hCudaWorld
				CudaCamera* hCudaCamera = nullptr;
				if (CudaWorld::m_hpm.GetSize() < sizeof(*hCudaCamera))
					ThrowException(L"insufficient host pinned memory for CudaCamera");
				hCudaCamera = (CudaCamera*)CudaWorld::m_hpm.GetPointerToMemory();

				CudaErrorCheck(cudaMemcpyAsync(
					hCudaCamera, &hCudaWorld->cameras[i],
					sizeof(CudaCamera),
					cudaMemcpyKind::cudaMemcpyDeviceToHost, mirror_stream));
				CudaErrorCheck(cudaStreamSynchronize(mirror_stream));

				if (!hCudaCamera->Exist()) continue;


				// [>] Asynchronous copying
				hCamera->m_samples_count = hCudaCamera->samples_count;

				static_assert(
					sizeof(*hCamera->GetBitmap().GetMapAddress()) ==
					sizeof(CudaColor<unsigned char>),
					"sizeof(Graphics::Color) != sizeof(CudaColor<unsigned char>)");

				// check cameras resolution
				if (hCamera->GetWidth() != hCudaCamera->width ||
					hCamera->GetHeight() != hCudaCamera->height) continue;

				uint32_t chunkSize =
					hCudaCamera->hostPinnedMemory.GetSize() /
					(sizeof(CudaColor<unsigned char>));
				if (chunkSize < 16u) ThrowException(L"Not enough host pinned memory for async image copy");

				uint32_t nPixels = hCamera->GetWidth() * hCamera->GetHeight();
				for (uint32_t startIndex = 0; startIndex < nPixels; startIndex += chunkSize)
				{
					// find start index
					if (startIndex + chunkSize > nPixels) chunkSize = nPixels - startIndex;

					// find offset point
					Graphics::Point<uint32_t> offset_point(startIndex % hCamera->GetWidth(), startIndex / hCamera->GetWidth());

					// copy final image data from hCudaCamera to hCudaPixels on pinned memory
					CudaColor<unsigned char>* hCudaPixels =
						(CudaColor<unsigned char>*)CudaCamera::hostPinnedMemory.GetPointerToMemory();
					CudaErrorCheck(cudaMemcpyFromArrayAsync(
						hCudaPixels, hCudaCamera->mp_final_image_array[m_update_ix],
						offset_point.x * sizeof(*hCudaPixels), offset_point.y,
						chunkSize * sizeof(*hCudaPixels),
						cudaMemcpyKind::cudaMemcpyDeviceToHost, mirror_stream));
					CudaErrorCheck(cudaStreamSynchronize(mirror_stream));

					// copy final image data from hostCudaPixels on pinned memory to hostCamera
					memcpy(hCamera->GetBitmap().GetMapAddress() + startIndex, hCudaPixels,
						chunkSize * sizeof(*hCudaPixels));
				}
			}
		}

		void Engine::LaunchFunction()
		{
			Timer function_timer, step_timer;

			while (true)
			{
				function_timer.Start();
				step_timer.Start();

				m_host_gate.Open();			// allow host to get things ready to render
				m_kernel_gate.WaitForOpen();// wait for host to prepare resources
				m_kernel_gate.Close();		// close gate for itself

				renderTimingString = L"Device side: \n";
				AppendTimeToString(renderTimingString, L"wait for host: ", step_timer.GetTime());

				if (m_launch_thread_terminate)
				{
					m_host_gate.Open();
					return;	// terminate launch function
				}


				// [>] Launch kernel for each camera
				for (uint32_t i = 0; i < m_launch_configs[m_render_ix].size(); ++i)
				{
					LaunchConfiguration& config = m_launch_configs[m_render_ix][i];
					cudaSetDevice(config.GetDeviceId());

					// [>] Update CudaCamera samples
					// reset samples values if needed
					step_timer.Start();
					if (config.GetUpdateFlag())
					{
						CudaKernel::CudaCameraSampleReset
							<< <
							config.GetGrid(),
							config.GetThreadBlock(),
							0u,
							m_render_stream
							>> >
							(mp_dCudaWorld, config.GetCameraId());
						CudaErrorCheck(cudaStreamSynchronize(m_render_stream));
						CudaErrorCheck(cudaGetLastError());
					}

					// increment samples number
					CudaKernel::CudaCameraUpdateSamplesNumber
						<< <
						1u, 1u, 0u, m_render_stream
						>> >
						(mp_dCudaWorld, config.GetCameraId(), config.GetUpdateFlag());
					CudaErrorCheck(cudaStreamSynchronize(m_render_stream));
					CudaErrorCheck(cudaGetLastError());
					AppendTimeToString(renderTimingString, L"update samples: ", step_timer.GetTime());


					// [>] Main render function
					step_timer.Start();
					CudaKernel::GenerateCameraRay
						<< <
						config.GetGrid(),
						config.GetThreadBlock(),
						config.GetSharedMemorySize(),
						m_render_stream
						>> >
						(/*mp_kernel_data[m_render_ix],*/
							mp_dCudaWorld,
							m_launch_configs[m_render_ix][i].GetCameraId());

					CudaErrorCheck(cudaStreamSynchronize(m_render_stream));
					CudaErrorCheck(cudaGetLastError());
					AppendTimeToString(renderTimingString, L"main render: ", step_timer.GetTime());


					// [>] Tone mapping
					step_timer.Start();
					CudaKernel::ToneMap
						<< <
						config.GetGrid(),
						config.GetThreadBlock(),
						0u,
						m_render_stream
						>> >
						(mp_kernel_data[m_render_ix],
							mp_dCudaWorld, config.GetCameraId());

					CudaErrorCheck(cudaStreamSynchronize(m_render_stream));
					CudaErrorCheck(cudaGetLastError());
					AppendTimeToString(renderTimingString, L"tone mapping: ", step_timer.GetTime());
					AppendTimeToString(renderTimingString, L"render full time: ", function_timer.GetTime());
				}
			}
		}
	}
}