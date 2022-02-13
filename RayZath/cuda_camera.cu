#include "cuda_camera.cuh"

namespace RayZath::Cuda
{
	__host__ TracingStates::TracingStates(const vec2ui32 resolution)
		: m_termination_counter(0u)
	{
		Resize(resolution);
	}
	__host__ void TracingStates::Resize(const vec2ui32 resolution)
	{
		m_path_depth.Reset(resolution);
		m_ray_origin.Reset(resolution);
		m_ray_direction.Reset(resolution);
		m_ray_material.Reset(resolution);
		m_ray_color.Reset(resolution);
		m_termination_index.Reset(resolution);
	}

	__host__ FrameBuffers::FrameBuffers(const vec2ui32 resolution)
	{
		Resize(resolution);
	}
	__host__ void FrameBuffers::Resize(const vec2ui32 resolution) 
	{
		for (size_t i = 0u; i < 2u; ++i)
		{
			m_image_buffer[i].Reset(resolution);
			m_depth_buffer[i].Reset(resolution);
		}
		m_space_buffer.Reset(resolution);
		m_final_image_buffer.Reset(resolution);
		m_final_depth_buffer.Reset(resolution);
	}
	
	
	HostPinnedMemory Camera::hostPinnedMemory(0x10000u);

	__host__ Camera::Camera(const vec2ui32 resolution)
		: resolution(resolution)
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
		, m_frame_buffers(resolution)
		, m_tracing_states(resolution)
	{}

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

			resolution = hCamera->GetResolution();
			m_frame_buffers.Resize(resolution);
			m_tracing_states.Resize(resolution);

			// resize hostPinnedMemory for mirroring
			this->hostPinnedMemory.SetMemorySize(
				std::min(
					resolution.x * resolution.y * uint32_t(sizeof(Color<unsigned char>)),
					0x100000u)); // max 1MiB
		}

		hCamera->GetStateRegister().MakeUnmodified();
	}
}