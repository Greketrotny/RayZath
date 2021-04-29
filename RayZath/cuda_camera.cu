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
			, fov(1.5f)
			, focal_distance(10.0f)
			, aperture(0.01f)
			, exposure_time(1.0f / 60.0f)
			, passes_count(0u)
			, inv_passes_count(1.0f)
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

			position = hCamera->GetPosition();
			rotation = hCamera->GetRotation();

			coord_system = hCamera->GetCoordSystem();

			aspect_ratio = hCamera->GetAspectRatio();
			fov = hCamera->GetFov().value();
			focal_distance = hCamera->GetFocalDistance();
			aperture = hCamera->GetAperture();
			exposure_time = hCamera->GetExposureTime();
			enabled = hCamera->Enabled();

			if (width != hCamera->GetWidth() || height != hCamera->GetHeight())
			{// resize buffers to match size of hostCamera resolution

				// destroy tracing paths
				if (mp_tracing_paths) CudaErrorCheck(cudaFree(mp_tracing_paths));


				// [>] Update CudaCamera resolution
				width = hCamera->GetWidth();
				height = hCamera->GetHeight();


				// [>] Reallocate resources
				// reset buffers
				m_sample_image_buffer.Reset(width, height);
				m_sample_depth_buffer.Reset(width, height);

				m_final_image_buffer[0].Reset(width, height);
				m_final_image_buffer[1].Reset(width, height);

				m_final_depth_buffer[0].Reset(width, height);
				m_final_depth_buffer[1].Reset(width, height);

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