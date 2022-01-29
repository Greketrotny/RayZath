#ifndef CUDA_CAMERA_CUH
#define CUDA_CAMERA_CUH

#include "rzexception.h"
#include "camera.h"
#include "cuda_engine_parts.cuh"
#include "cuda_render_parts.cuh"
#include "cuda_buffer.cuh"

namespace RayZath::Cuda
{
	class World;

	template <typename T>
	struct SOA;	// Structure Of Arrays. Specializations should contain separate buffer for each T component.

	struct TracingState
	{
		ColorF final_color;
		uint8_t path_depth;

		__device__ TracingState(const ColorF color, const uint8_t depth)
			: final_color(color)
			, path_depth(depth)
		{}


		__device__ bool NextNodeAvailable()
		{
			return path_depth != 8u - 1u;
		}
		__device__ bool FindNextNodeToTrace()
		{
			return path_depth++ != 8u - 1u;
		}
		__device__ void EndPath()
		{
			path_depth = 8u - 1u;
		}
	};

	struct TracingStates
	{
	private:
		GlobalBuffer<uint8_t> m_path_depth;

		GlobalBuffer<vec3f> m_ray_origin;
		GlobalBuffer<vec3f> m_ray_direction;
		GlobalBuffer<Material*> m_ray_material;
		GlobalBuffer<ColorF> m_ray_color;

	public:
		__host__ TracingStates(const vec2ui32 resolution);

	public:
		__host__ void Resize(const vec2ui32 resolution);
	};

	struct FrameBuffers
	{
	private:
		SurfaceBuffer<ColorF> m_sample_image_buffer[2];
		SurfaceBuffer<float> m_sample_depth_buffer[2];
		SurfaceBuffer<uint16_t> m_passes_buffer[2];
		SurfaceBuffer<ColorU> m_final_image_buffer;
		SurfaceBuffer<float> m_final_depth_buffer;
		SurfaceBuffer<vec3f> m_space_buffer;

	public:
		__host__ FrameBuffers(const vec2ui32 resolution);

	public:
		__host__ void Resize(const vec2ui32 resolution);

		__device__ __inline__ SurfaceBuffer<ColorF>& SampleImageBuffer(const bool idx)
		{
			return m_sample_image_buffer[uint8_t(idx)];
		}
		__device__ __inline__ SurfaceBuffer<float>& SampleDepthBuffer(const bool idx)
		{
			return m_sample_depth_buffer[uint8_t(idx)];
		}
		__device__ __inline__ SurfaceBuffer<uint16_t>& PassesBuffer(const bool idx)
		{
			return m_passes_buffer[uint8_t(idx)];
		}
		__host__ __device__ __inline__ SurfaceBuffer<ColorU>& FinalImageBuffer()
		{
			return m_final_image_buffer;
		}
		__host__ __device__ __inline__ SurfaceBuffer<float>& FinalDepthBuffer()
		{
			return m_final_depth_buffer;
		}
		__device__ __inline__ SurfaceBuffer<vec3f>& SpaceBuffer()
		{
			return m_space_buffer;
		}
	};

	class Camera
	{
	private:
		vec3f position[2];
		CoordSystem coord_system[2];

		vec2ui32 resolution;
		float aspect_ratio;
		bool enabled;
		float fov[2];
		vec2f near_far;

		float focal_distance;
		float aperture;
		float exposure_time;
		float temporal_blend;

		uint32_t passes_count;
		bool sample_buffer_idx;

		FrameBuffers m_frame_buffers;
		TracingStates m_tracing_states;

	public:
		static HostPinnedMemory hostPinnedMemory;


	public:
		__host__ Camera(const vec2ui32 resolution = { 8u, 8u });


		__host__ void Reconstruct(
			const World& hCudaWorld,
			const RayZath::Engine::Handle<RayZath::Engine::Camera>& hCamera,
			cudaStream_t& mirror_stream);
	public:
		__host__ __device__ vec3f& CurrentPosition()
		{
			return position[0];
		}
		__host__ __device__ vec3f& PreviousPosition()
		{
			return position[1];
		}
		__host__ __device__ CoordSystem& CurrentCoordSystem()
		{
			return coord_system[0];
		}
		__host__ __device__ CoordSystem& PreviousCoordSystem()
		{
			return coord_system[1];
		}
		__host__ __device__ float& CurrentFov()
		{
			return fov[0];
		}
		__host__ __device__ float& PreviousFov()
		{
			return fov[1];
		}
		__host__ __device__ vec2f GetNearFar()
		{
			return near_far;
		}
		__host__ __device__ uint32_t GetWidth() const
		{
			return resolution.x;
		}
		__host__ __device__ uint32_t GetHeight() const
		{
			return resolution.y;
		}
		__host__ __device__ vec2ui32 GetResolution() const
		{
			return resolution;
		}
		__host__ __device__ uint32_t GetPassesCount() const
		{
			return passes_count;
		}
		__host__ __device__ uint32_t& GetPassesCount()
		{
			return passes_count;
		}
		__device__ float GetAperture() const
		{
			return aperture;
		}
		__device__ float GetExposureTime() const
		{
			return exposure_time;
		}

		__device__ __inline__ void SwapImageBuffers()
		{
			sample_buffer_idx = !sample_buffer_idx;
		}
		__device__ __inline__ SurfaceBuffer<ColorF>& SampleImageBuffer()
		{
			return m_frame_buffers.SampleImageBuffer(sample_buffer_idx);
		}
		__device__ __inline__ SurfaceBuffer<ColorF>& EmptyImageBuffer()
		{
			return m_frame_buffers.SampleImageBuffer(!sample_buffer_idx);
		}
		__device__ __inline__ SurfaceBuffer<float>& CurrentDepthBuffer()
		{
			return m_frame_buffers.SampleDepthBuffer(sample_buffer_idx);
		}
		__device__ __inline__ SurfaceBuffer<float>& PreviousDepthBuffer()
		{
			return m_frame_buffers.SampleDepthBuffer(!sample_buffer_idx);
		}
		__host__ __device__ __inline__ auto& FinalImageBuffer()
		{
			return m_frame_buffers.FinalImageBuffer();
		}
		__host__ __device__ __inline__ SurfaceBuffer<float>& FinalDepthBuffer()
		{
			return m_frame_buffers.FinalDepthBuffer();
		}
		__device__ __inline__ SurfaceBuffer<vec3f>& SpaceBuffer()
		{
			return m_frame_buffers.SpaceBuffer();
		}
		__device__ __inline__ SurfaceBuffer<uint16_t>& PassesBuffer()
		{
			return m_frame_buffers.PassesBuffer(sample_buffer_idx);
		}
		__device__ __inline__ SurfaceBuffer<uint16_t>& EmptyPassesBuffer()
		{
			return m_frame_buffers.PassesBuffer(!sample_buffer_idx);
		}


		// ray generation
	public:
		__device__ void GenerateSimpleRay(
			SceneRay& ray,
			FullThread& thread,
			RNG& rng)
		{
			ray.direction = vec3f(0.0f, 0.0f, 1.0f);
			ray.origin = vec3f(0.0f);

			// ray to screen deflection
			const float tana = cui_tanf(CurrentFov() * 0.5f);
			const vec2f dir =
				(((vec2f(thread.in_grid) + vec2f(0.5f)) /
					vec2f(resolution)) -
					vec2f(0.5f)) *
				vec2f(tana, -tana / aspect_ratio);
			ray.direction.x = dir.x;
			ray.direction.y = dir.y;

			// pixel position distortion (antialiasing)
			ray.direction.x +=
				((0.5f / float(resolution.x)) * (rng.SignedUniform()));
			ray.direction.y +=  // this --v-- should be x
				((0.5f / float(resolution.x)) * (rng.SignedUniform()));

			// camera transformation
			CurrentCoordSystem().TransformBackward(ray.origin);
			CurrentCoordSystem().TransformBackward(ray.direction);
			ray.direction.Normalize();
			ray.origin += CurrentPosition();

			// apply near/far clipping plane
			ray.near_far = near_far;
		}
		__device__ void GenerateRay(
			SceneRay& ray,
			FullThread& thread,
			RNG& rng)
		{
			ray.direction = vec3f(0.0f, 0.0f, 1.0f);

			// ray to screen deflection
			const float tana = cui_tanf(CurrentFov() * 0.5f);
			const vec2f dir =
				(((vec2f(thread.in_grid) + vec2f(0.5f)) /
					vec2f(resolution)) -
					vec2f(0.5f)) *
				vec2f(tana, -tana / aspect_ratio);
			ray.direction.x = dir.x;
			ray.direction.y = dir.y;

			// pixel position distortion (antialiasing)
			ray.direction.x +=
				((0.5f / float(resolution.x)) * (rng.SignedUniform()));
			ray.direction.y +=  // this --v-- should be x
				((0.5f / float(resolution.x)) * (rng.SignedUniform()));

			// focal point
			const vec3f focalPoint = ray.direction * focal_distance;

			// aperture distortion
			const float apertureAngle = rng.UnsignedUniform() * CUDART_PI_F * 2.0f;
			const float apertureSample = sqrtf(rng.UnsignedUniform()) * aperture;
			ray.origin += vec3f(
				apertureSample * cui_sinf(apertureAngle),
				apertureSample * cui_cosf(apertureAngle),
				0.0f);

			// depth of field ray
			ray.direction = focalPoint - ray.origin;

			// camera transformation
			CurrentCoordSystem().TransformBackward(ray.origin);
			CurrentCoordSystem().TransformBackward(ray.direction);
			ray.direction.Normalize();
			ray.origin += CurrentPosition();

			// apply near/far clipping plane
			ray.near_far = near_far;
		}


		// Spatio-temporal reprojection
	private:
		template <typename T>
		__device__ T Blend(const T v1, const T v2, const float a)
		{
			return (v1 * a) + (v2 * (1.0f - a));
		}
	public:
		__device__ void Reproject(
			const vec2ui32 pixel)
		{
			// get spatial point
			const vec3f space_p = SpaceBuffer().GetValue(pixel);
			// transform point to local camera space
			vec3f local_p = space_p - PreviousPosition();
			PreviousCoordSystem().TransformForward(local_p);

			if (local_p.z <= 0.0f)
				return;	// point is behind camera screen

			// project point on camera screen
			const float tana = cui_tanf(PreviousFov() * 0.5f);
			const vec2f screen_p =
				(((vec2f(local_p.x, local_p.y) /
					local_p.z) /
					vec2f(tana, -tana / aspect_ratio)) +
					vec2f(0.5f)) *
				vec2f(resolution);

			if (screen_p.x < 0.0f || screen_p.x >= resolution.x ||
				screen_p.y < 0.0f || screen_p.y >= resolution.y)
				return;	// projected point falls outside camera frustum


			// compare stored depth values
			const float point_dist = vec3f::Distance(PreviousPosition(), space_p);
			const float buffer_dist = PreviousDepthBuffer().GetValue(vec2ui32(screen_p));
			const float delta_dist = point_dist - buffer_dist;
			if (fabsf(delta_dist) < 0.01f * point_dist)
			{
				SampleImageBuffer().SetValue(
					pixel,
					Blend(EmptyImageBuffer().GetValue(vec2ui32(screen_p)) /
						EmptyPassesBuffer().GetValue(vec2ui32(screen_p)),
						SampleImageBuffer().GetValue(pixel),
						temporal_blend));
			}
		}
	};
}

#endif // !CUDA_CAMERA_CUH