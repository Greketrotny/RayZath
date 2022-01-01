#include "cuda_camera.cuh"

namespace RayZath::Cuda
{
	HostPinnedMemory Camera::hostPinnedMemory(0x10000u);

	__host__ Camera::Camera()
		: resolution(0u, 0u)
		, aspect_ratio(1.0f)
		, enabled(true)
		, fov{ 1.5f, 1.5f }
		, near_far(0.01f, 1000.0f)
		, focal_distance(10.0f)
		, aperture(0.01f)
		, exposure_time(1.0f / 60.0f)
		, temporal_blend(0.75f)
		, passes_count(0u)
		, sample_buffer_idx(false)
		, mp_tracing_paths(nullptr)
	{}
	__host__ Camera::~Camera()
	{
		// destroy tracing paths
		if (mp_tracing_paths) CudaErrorCheck(cudaFree(mp_tracing_paths));
		mp_tracing_paths = nullptr;
	}

	__host__ void Camera::Reconstruct(
		const World& hCudaWorld,
		const RayZath::Engine::Handle<RayZath::Engine::Camera>& hCamera,
		cudaStream_t& mirror_stream)
	{
		if (!hCamera->GetStateRegister().IsModified()) return;

		PreviousPosition() = CurrentPosition();
		CurrentPosition() = hCamera->GetPosition();

		PreviousCoordSystem() = CurrentCoordSystem();
		CurrentCoordSystem() = hCamera->GetCoordSystem();

		enabled = hCamera->Enabled();
		aspect_ratio = hCamera->GetAspectRatio();
		PreviousFov() = CurrentFov();
		CurrentFov() = hCamera->GetFov().value();
		near_far = hCamera->GetNearFar();
		focal_distance = hCamera->GetFocalDistance();
		aperture = hCamera->GetAperture();
		exposure_time = hCamera->GetExposureTime();
		temporal_blend = hCamera->GetTemporalBlend();

		if (resolution != hCamera->GetResolution())
		{// resize buffers to match size of hostCamera resolution

			// destroy tracing paths
			if (mp_tracing_paths) CudaErrorCheck(cudaFree(mp_tracing_paths));


			// [>] Update Camera resolution
			resolution = hCamera->GetResolution();


			// [>] Reallocate resources
			// reset buffers
			m_sample_image_buffer[0].Reset(resolution);
			m_sample_image_buffer[1].Reset(resolution);

			m_sample_depth_buffer[0].Reset(resolution);
			m_sample_depth_buffer[1].Reset(resolution);

			m_final_image_buffer.Reset(resolution);
			m_final_depth_buffer.Reset(resolution);

			m_space_buffer.Reset(resolution);
			m_passes_buffer[0].Reset(resolution);
			m_passes_buffer[1].Reset(resolution);

			// allocate memory for tracing paths
			CudaErrorCheck(cudaMalloc(
				(void**)&mp_tracing_paths,
				size_t(resolution.x) * size_t(resolution.y) * size_t(sizeof(*mp_tracing_paths))));


			// [>] Resize hostPinnedMemory for mirroring
			this->hostPinnedMemory.SetMemorySize(
				std::min(
					resolution.x * resolution.y * uint32_t(sizeof(Color<unsigned char>)),
					0x100000u)); // max 1MiB
		}

		hCamera->GetStateRegister().MakeUnmodified();
	}
}