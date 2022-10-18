#ifndef CUDA_CAMERA_CUH
#define CUDA_CAMERA_CUH

#include "rzexception.hpp"
#include "camera.hpp"

#include "cuda_engine_parts.cuh"
#include "cuda_render_parts.cuh"
#include "cuda_buffer.cuh"

namespace RayZath::Cuda
{
	class World;

	struct TracingState
	{
		ColorF final_color;
		uint8_t path_depth;
		static constexpr uint8_t sm_path_limit = std::numeric_limits<decltype(path_depth)>::max();

		__device__ TracingState(const ColorF color, const uint8_t depth)
			: final_color(color)
			, path_depth(depth)
		{}

		__device__ void endPath()
		{
			path_depth = sm_path_limit;
		}
	};

	struct TracingStates
	{
	private:
		GlobalBuffer<uint8_t> m_path_depth;

		GlobalBuffer<vec3f> m_ray_origin;
		GlobalBuffer<vec3f> m_ray_direction;
		GlobalBuffer<const Material*> m_ray_material;
		GlobalBuffer<ColorF> m_ray_color;

	public:
		__host__ TracingStates(const vec2ui32 resolution);

	public:
		__host__ void resize(const vec2ui32 resolution);

		__device__ __inline__ void setRay(const vec2ui32 pixel, const SceneRay& ray)
		{
			m_ray_origin.SetValue(pixel, ray.origin);
			m_ray_direction.SetValue(pixel, ray.direction);
			m_ray_material.SetValue(pixel, ray.material);
			m_ray_color.SetValue(pixel, ray.color);
		}
		__device__ __inline__ SceneRay getRay(const vec2ui32 pixel)
		{
			SceneRay ray;
			ray.origin = m_ray_origin.GetValue(pixel);
			ray.direction = m_ray_direction.GetValue(pixel);
			ray.material = m_ray_material.GetValue(pixel);
			ray.color = m_ray_color.GetValue(pixel);
			return ray;
		}

		__device__ __inline__ uint8_t getPathDepth(const vec2ui32 pixel)
		{
			return m_path_depth.GetValue(pixel);
		}
		__device__ __inline__ void setPathDepth(const vec2ui32 pixel, const uint8_t depth)
		{
			m_path_depth.SetValue(pixel, depth);
		}
	};

	struct FrameBuffers
	{
	private:
		SurfaceBuffer<ColorF> m_image_buffer[2];
		SurfaceBuffer<float> m_depth_buffer[2];
		SurfaceBuffer<vec3f> m_space_buffer;
		SurfaceBuffer<ColorU> m_final_image_buffer;
		SurfaceBuffer<float> m_final_depth_buffer;

	public:
		__host__ FrameBuffers(const vec2ui32 resolution);

	public:
		__host__ void resize(const vec2ui32 resolution);

		__device__ __inline__ SurfaceBuffer<ColorF>& sampleImageBuffer(const bool idx)
		{
			return m_image_buffer[uint8_t(idx)];
		}
		__device__ __inline__ SurfaceBuffer<float>& sampleDepthBuffer(const bool idx)
		{
			return m_depth_buffer[uint8_t(idx)];
		}
		__device__ __inline__ SurfaceBuffer<vec3f>& spaceBuffer()
		{
			return m_space_buffer;
		}
		__host__ __device__ __inline__ SurfaceBuffer<ColorU>& finalImageBuffer()
		{
			return m_final_image_buffer;
		}
		__host__ __device__ __inline__ SurfaceBuffer<float>& finalDepthBuffer()
		{
			return m_final_depth_buffer;
		}
	};

	class Camera
	{
	private:
		vec3f position[2];
		CoordSystem coord_system[2];

		vec2ui32 m_resolution = vec2ui32(640u, 360u);
		float aspect_ratio = m_resolution.x / float(m_resolution.y);

		float fov[2] = { 1.5f, 1.5f };
		vec2f near_far = vec2f(0.01f, 1000.0f);
		float focal_distance = 1.0f;
		float m_aperture = 0.01f;
		float exposure_time = 1.0f / 60.0f;
		float temporal_blend = 0.75f;

		uint32_t passes_count[2] = { 0u, 0u };
		uint64_t m_ray_count[2] = { 0u, 0u };
		bool m_current_idx = false;
		static constexpr bool m_render_idx = false;

		FrameBuffers m_frame_buffers;
		TracingStates m_tracing_states;

		vec2ui32 m_ray_cast_pixel;
	public:
		uint32_t m_instance_idx, m_instance_material_idx;

		static HostPinnedMemory hostPinnedMemory;


	public:
		__host__ Camera(const vec2ui32 resolution = { 640u, 360u });


		__host__ void reconstruct(
			const World& hCudaWorld,
			const RayZath::Engine::Handle<RayZath::Engine::Camera>& hCamera,
			cudaStream_t& mirror_stream);
	public:
		__host__ void swapHistoryIdx()
		{
			m_current_idx = !m_current_idx;
		}
		__host__ __device__ bool getCurrentIdx() const
		{
			return m_current_idx;
		}
		__host__ __device__ bool getPreviousIdx() const
		{
			return !m_current_idx;
		}
		__host__ __device__ bool renderIdx() const
		{
			return m_render_idx;
		}
		__host__ __device__ bool getResultIdx() const
		{
			return !m_render_idx;	
		}

		__host__ __device__ vec3f& currentPosition()
		{
			return position[getCurrentIdx()];
		}
		__host__ __device__ vec3f& previousPosition()
		{
			return position[getPreviousIdx()];
		}
		__host__ __device__ CoordSystem& currentCoordSystem()
		{
			return coord_system[getCurrentIdx()];
		}
		__host__ __device__ CoordSystem& previousCoordSystem()
		{
			return coord_system[getPreviousIdx()];
		}

		__host__ __device__ uint32_t width() const
		{
			return m_resolution.x;
		}
		__host__ __device__ uint32_t height() const
		{
			return m_resolution.y;
		}
		__host__ __device__ vec2ui32 resolution() const
		{
			return m_resolution;
		}

		__host__ __device__ float& currentFov()
		{
			return fov[getCurrentIdx()];
		}
		__host__ __device__ float& previousFov()
		{
			return fov[getPreviousIdx()];
		}
		__host__ __device__ vec2f nearFar()
		{
			return near_far;
		}
		__device__ float aperture() const
		{
			return m_aperture;
		}
		__device__ float apertureArea() const
		{
			return CUDART_PI_F * aperture() * aperture();
		}
		__device__ float exposureTime() const
		{
			return exposure_time;
		}

		__host__ __device__ auto getRenderPassCount() const
		{
			return passes_count[renderIdx()];
		}
		__host__ __device__ auto getResultPassCount() const
		{
			return passes_count[getResultIdx()];
		}
		__device__ void setRenderPassCount(uint32_t count)
		{
			passes_count[renderIdx()] = count;
		}
		__device__ void setResultPassCount(uint32_t count)
		{
			passes_count[getResultIdx()] = count;
		}
		
		__host__ __device__ auto getRenderRayCount() const
		{
			return m_ray_count[renderIdx()];
		}
		__host__ __device__ auto getResultRayCount() const
		{
			return m_ray_count[getResultIdx()];
		}
		__device__ void setRenderRayCount(uint64_t count)
		{
			m_ray_count[renderIdx()] = count;
		}
		__device__ void setResultRayCount(uint64_t count)
		{
			m_ray_count[getResultIdx()] = count;
		}
				
		__device__ auto& getTracingStates()
		{
			return m_tracing_states;
		}
		__device__ auto getRayCastPixel() const
		{
			return m_ray_cast_pixel;
		}

		__device__ __inline__ SurfaceBuffer<ColorF>& currentImageBuffer()
		{
			return m_frame_buffers.sampleImageBuffer(m_current_idx);
		}
		__device__ __inline__ SurfaceBuffer<ColorF>& previousImageBuffer()
		{
			return m_frame_buffers.sampleImageBuffer(!m_current_idx);
		}
		__device__ __inline__ SurfaceBuffer<float>& currentDepthBuffer()
		{
			return m_frame_buffers.sampleDepthBuffer(m_current_idx);
		}
		__device__ __inline__ SurfaceBuffer<float>& previousDepthBuffer()
		{
			return m_frame_buffers.sampleDepthBuffer(!m_current_idx);
		}
		__device__ __inline__ SurfaceBuffer<vec3f>& spaceBuffer()
		{
			return m_frame_buffers.spaceBuffer();
		}
		__host__ __device__ __inline__ auto& finalImageBuffer()
		{
			return m_frame_buffers.finalImageBuffer();
		}
		__host__ __device__ __inline__ SurfaceBuffer<float>& finalDepthBuffer()
		{
			return m_frame_buffers.finalDepthBuffer();
		}


		// ray generation
	public:
		__device__ void generateSimpleRay(
			RangedRay& ray,
			const vec2f pos)
		{
			ray.origin = vec3f(0.0f);

			// ray to screen deflection
			const float tana = cui_tanf(currentFov() * 0.5f);
			const vec2f dir =
				(((pos + vec2f(0.5f)) /
					vec2f(resolution())) -
					vec2f(0.5f)) *
				vec2f(tana, -tana / aspect_ratio);
			ray.direction.x = dir.x;
			ray.direction.y = dir.y;
			ray.direction.z = 1.0f;

			// camera transformation
			currentCoordSystem().transformBackward(ray.origin);
			currentCoordSystem().transformBackward(ray.direction);
			ray.direction.Normalize();
			ray.origin += currentPosition();

			// apply near/far clipping plane
			ray.near_far = near_far;
		}
		__device__ void generateSimpleRay(
			RangedRay& ray,
			const GridThread& thread)
		{
			return generateSimpleRay(ray, vec2f(thread.grid_pos));
		}
		__device__ void generateRay(
			RangedRay& ray,
			const vec2ui32 pixel,
			RNG& rng)
		{
			// ray to screen deflection
			const float tana = cui_tanf(currentFov() * 0.5f);
			const vec2f dir =
				(((vec2f(pixel) + vec2f(0.5f)) /
					vec2f(m_resolution)) -
					vec2f(0.5f)) *
				vec2f(tana, -tana / aspect_ratio);
			ray.direction.x = dir.x;
			ray.direction.y = dir.y;
			ray.direction.z = 1.0f;

			// pixel position distortion (antialiasing)
			ray.direction.x +=
				((0.5f / float(m_resolution.x)) * (rng.SignedUniform()));
			ray.direction.y +=  // this --v-- should be x
				((0.5f / float(m_resolution.x)) * (rng.SignedUniform()));

			// focal point
			const vec3f focalPoint = ray.direction * focal_distance;

			// aperture distortion
			const float apertureAngle = rng.unsignedUniform() * CUDART_PI_F * 2.0f;
			const float apertureSample = sqrtf(rng.unsignedUniform()) * aperture();
			ray.origin = vec3f(
				apertureSample * cui_sinf(apertureAngle),
				apertureSample * cui_cosf(apertureAngle),
				0.0f);

			// depth of field ray
			ray.direction = focalPoint - ray.origin;

			// camera transformation
			currentCoordSystem().transformBackward(ray.origin);
			ray.origin += currentPosition();
			currentCoordSystem().transformBackward(ray.direction);
			ray.direction.Normalize();

			// apply near/far clipping plane
			ray.near_far = near_far;
		}


		// Spatio-temporal reprojection
	private:
		template <typename T>
		__device__ T blend(const T v1, const T v2, const float a)
		{
			return (v1 * a) + (v2 * (1.0f - a));
		}
	public:
		__device__ void reproject(
			const vec2ui32 to_pixel)
		{
			// get current spatial point
			const vec3f space_p = spaceBuffer().GetValue(to_pixel);
			// transform point to previous local camera space
			vec3f local_p = space_p - previousPosition();
			previousCoordSystem().transformForward(local_p);

			if (local_p.z <= 0.0f)
				return;	// point is behind camera screen

			// project point on camera screen
			const float tana = cui_tanf(previousFov() * 0.5f);
			const vec2f from_pixel =
				(((vec2f(local_p.x, local_p.y) /
					local_p.z) /
					vec2f(tana, -tana / aspect_ratio)) +
					vec2f(0.5f)) *
				vec2f(m_resolution);

			if (from_pixel.x < 0.0f || from_pixel.x >= m_resolution.x ||
				from_pixel.y < 0.0f || from_pixel.y >= m_resolution.y)
				return;	// projected point falls outside previous camera frustum


			// compare stored depth values
			const float point_dist = vec3f::Distance(previousPosition(), space_p);
			const float buffer_dist = previousDepthBuffer().GetValue(vec2ui32(from_pixel));
			const float delta_dist = point_dist - buffer_dist;
			if (fabsf(delta_dist) < 0.01f * point_dist)
			{
				currentImageBuffer().AppendValue(
					to_pixel,
					previousImageBuffer().GetValue(vec2ui32(from_pixel)) * temporal_blend);
			}
		}
	};
}

#endif // !CUDA_CAMERA_CUH