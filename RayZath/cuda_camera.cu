#include "cuda_camera.cuh"

#include "world.hpp"
#include "camera.hpp"

namespace RayZath::Cuda
{
	__host__ TracingStates::TracingStates(const vec2ui32 resolution)
	{
		resize(resolution);
	}
	__host__ void TracingStates::resize(const vec2ui32 resolution)
	{
		m_path_depth.reset(resolution);
		m_ray_origin.reset(resolution);
		m_ray_direction.reset(resolution);
		m_ray_material.reset(resolution);
		m_ray_color.reset(resolution);
	}

	__host__ FrameBuffers::FrameBuffers(const vec2ui32 resolution)
	{
		resize(resolution);
	}
	__host__ void FrameBuffers::resize(const vec2ui32 resolution)
	{
		for (size_t i = 0u; i < 2u; ++i)
		{
			m_image_buffer[i].reset(resolution);
			m_depth_buffer[i].reset(resolution);
		}
		m_space_buffer.reset(resolution);
		m_final_image_buffer.reset(resolution);
		m_final_depth_buffer.reset(resolution);
	}


	HostPinnedMemory Camera::hostPinnedMemory(0x10000u);

	__host__ Camera::Camera(const vec2ui32 resolution)
		: m_resolution(resolution)
		, m_frame_buffers(resolution)
		, m_tracing_states(resolution)
		, m_instance_idx{std::numeric_limits<uint32_t>::max()}
		, m_instance_material_idx{std::numeric_limits<uint32_t>::max()}
	{}

	__host__ void Camera::reconstruct(
		[[maybe_unused]] const World& hCudaWorld,
		const RayZath::Engine::Handle<RayZath::Engine::Camera>& hCamera,
		[[maybe_unused]] cudaStream_t& mirror_stream)
	{
		if (!hCamera->stateRegister().IsModified()) return;

		swapHistoryIdx();

		currentPosition() = hCamera->position();
		currentCoordSystem() = hCamera->coordSystem();

		aspect_ratio = hCamera->aspectRatio();
		currentFov() = hCamera->fov().value();
		near_far = hCamera->nearFar();
		focal_distance = hCamera->focalDistance();
		m_aperture = hCamera->aperture();
		exposure_time = hCamera->exposureTime();
		temporal_blend = hCamera->temporalBlend();
		m_ray_cast_pixel = hCamera->getRayCastPixel();

		if (m_resolution != hCamera->resolution())
		{// resize buffers to match size of hostCamera resolution

			m_resolution = hCamera->resolution();
			m_frame_buffers.resize(m_resolution);
			m_tracing_states.resize(m_resolution);

			// resize hostPinnedMemory for mirroring
			this->hostPinnedMemory.SetMemorySize(
				std::min(
					m_resolution.x * m_resolution.y * uint32_t(sizeof(Color<unsigned char>)),
					0x100000u)); // max 1MiB
		}

		hCamera->stateRegister().MakeUnmodified();
	}
}