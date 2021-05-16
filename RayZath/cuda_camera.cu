#include "cuda_camera.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		HostPinnedMemory CudaCamera::hostPinnedMemory(0x10000u);

		__host__ CudaCamera::CudaCamera()
			: width(0u), height(0u)
			, aspect_ratio(1.0f)
			, enabled(true)
			, fov{1.5f, 1.5f}
			, focal_distance(10.0f)
			, aperture(0.01f)
			, exposure_time(1.0f / 60.0f)
			, temporal_blend(0.75f)
			, passes_count(0u)
			, sample_buffer_idx(false)
			, mp_tracing_paths(nullptr)
		{}
		__host__ CudaCamera::~CudaCamera()
		{
			// destroy tracing paths
			if (mp_tracing_paths) CudaErrorCheck(cudaFree(mp_tracing_paths));
			mp_tracing_paths = nullptr;
		}

		__host__ void CudaCamera::Reconstruct(
			const CudaWorld& hCudaWorld, 
			const Handle<Camera>& hCamera, 
			cudaStream_t& mirror_stream)
		{
			if (!hCamera->GetStateRegister().IsModified()) return;
			
			PreviousPosition() = CurrentPosition();
			CurrentPosition() = hCamera->GetPosition();
			PreviousRotation() = CurrentRotation();
			CurrentRotation() = hCamera->GetRotation();

			PreviousCoordSystem() = CurrentCoordSystem();
			CurrentCoordSystem() = hCamera->GetCoordSystem();

			enabled = hCamera->Enabled();
			aspect_ratio = hCamera->GetAspectRatio();
			PreviousFov() = CurrentFov();
			CurrentFov() = hCamera->GetFov().value();
			focal_distance = hCamera->GetFocalDistance();
			aperture = hCamera->GetAperture();
			exposure_time = hCamera->GetExposureTime();
			temporal_blend = hCamera->GetTemporalBlend();

			if (width != hCamera->GetWidth() || height != hCamera->GetHeight())
			{// resize buffers to match size of hostCamera resolution

				// destroy tracing paths
				if (mp_tracing_paths) CudaErrorCheck(cudaFree(mp_tracing_paths));


				// [>] Update CudaCamera resolution
				width = hCamera->GetWidth();
				height = hCamera->GetHeight();


				// [>] Reallocate resources
				// reset buffers
				m_sample_image_buffer[0].Reset(width, height);
				m_sample_image_buffer[1].Reset(width, height);

				m_sample_depth_buffer[0].Reset(width, height);
				m_sample_depth_buffer[1].Reset(width, height);

				m_final_image_buffer[0].Reset(width, height);
				m_final_image_buffer[1].Reset(width, height);

				m_final_depth_buffer[0].Reset(width, height);
				m_final_depth_buffer[1].Reset(width, height);

				m_space_buffer.Reset(width, height);
				m_passes_buffer[0].Reset(width, height);
				m_passes_buffer[1].Reset(width, height);

				// allocate memory for tracing paths
				CudaErrorCheck(cudaMalloc(
					(void**)&mp_tracing_paths, 
					size_t(width) * size_t(height) * size_t(sizeof(*mp_tracing_paths))));


				// [>] Resize hostPinnedMemory for mirroring
				this->hostPinnedMemory.SetMemorySize(
					std::min(
						width * height * uint32_t(sizeof(Color<unsigned char>)),
						0x100000u)); // max 1MiB
			}

			hCamera->GetStateRegister().MakeUnmodified();
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	}
}