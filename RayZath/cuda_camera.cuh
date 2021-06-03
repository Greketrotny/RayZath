#ifndef CUDA_CAMERA_CUH
#define CUDA_CAMERA_CUH

#include "rzexception.h"
#include "camera.h"
#include "exist_flag.cuh"
#include "cuda_engine_parts.cuh"
#include "cuda_render_parts.cuh"
#include "cuda_buffer.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		class CudaWorld;

		class CudaCamera : public WithExistFlag
		{
		private:
			vec3f position[2];
			vec3f rotation[2];	// TODO: Try to remove angular rotation
			CudaCoordSystem coord_system[2];

			vec2ui32 resolution;
			float aspect_ratio;
			bool enabled;
			float fov[2];

			float focal_distance;
			float aperture;
			float exposure_time;
			float temporal_blend;

			uint32_t passes_count;
			bool sample_buffer_idx;

			CudaSurfaceBuffer<ColorF> m_sample_image_buffer[2];
			CudaSurfaceBuffer<float> m_sample_depth_buffer[2];
			CudaSurfaceBuffer<ColorU> m_final_image_buffer;
			CudaSurfaceBuffer<float> m_final_depth_buffer;

			CudaSurfaceBuffer<vec3f> m_space_buffer;
			CudaSurfaceBuffer<uint16_t> m_passes_buffer[2];

			TracingPath* mp_tracing_paths;
		public:
			static HostPinnedMemory hostPinnedMemory;


		public:
			__host__ CudaCamera();
			__host__ ~CudaCamera();


			__host__ void Reconstruct(
				const CudaWorld& hCudaWorld,
				const Handle<Camera>& hCamera,
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
			__host__ __device__ vec3f& CurrentRotation()
			{
				return rotation[0];
			}
			__host__ __device__ vec3f& PreviousRotation()
			{
				return rotation[1];
			}
			__host__ __device__ CudaCoordSystem& CurrentCoordSystem()
			{
				return coord_system[0];
			}
			__host__ __device__ CudaCoordSystem& PreviousCoordSystem()
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
			__host__ __device__ const uint32_t& GetWidth() const
			{
				return resolution.x;
			}
			__host__ __device__ const uint32_t& GetHeight() const
			{
				return resolution.y;
			}
			__host__ __device__ const vec2ui32& GetResolution() const
			{
				return resolution;
			}
			__host__ __device__ const uint32_t& GetPassesCount() const
			{
				return passes_count;
			}
			__host__ __device__ uint32_t& GetPassesCount()
			{
				return passes_count;
			}
			__device__ const float& GetAperture() const
			{
				return aperture;
			}
			__device__ const float& GetExposureTime() const
			{
				return exposure_time;
			}

			__device__ __inline__ CudaSurfaceBuffer<ColorF>& SampleImageBuffer()
			{
				return m_sample_image_buffer[sample_buffer_idx];
			}
			__device__ __inline__ CudaSurfaceBuffer<ColorF>& EmptyImageBuffer()
			{
				return m_sample_image_buffer[!sample_buffer_idx];
			}
			__device__ __inline__ void SwapImageBuffers()
			{
				sample_buffer_idx = !sample_buffer_idx;
			}

			__device__ __inline__ CudaSurfaceBuffer<float>& CurrentDepthBuffer()
			{
				return m_sample_depth_buffer[sample_buffer_idx];
			}
			__device__ __inline__ CudaSurfaceBuffer<float>& PreviousDepthBuffer()
			{
				return m_sample_depth_buffer[!sample_buffer_idx];
			}
			__host__ __device__ __inline__ CudaSurfaceBuffer<ColorU>& FinalImageBuffer()
			{
				return m_final_image_buffer;
			}
			__host__ __device__ __inline__ CudaSurfaceBuffer<float>& FinalDepthBuffer()
			{
				return m_final_depth_buffer;
			}

			__device__ __inline__ CudaSurfaceBuffer<vec3f>& SpaceBuffer()
			{
				return m_space_buffer;
			}
			__device__ __inline__ CudaSurfaceBuffer<uint16_t>& PassesBuffer()
			{
				return m_passes_buffer[sample_buffer_idx];
			}
			__device__ __inline__ CudaSurfaceBuffer<uint16_t>& EmptyPassesBuffer()
			{
				return m_passes_buffer[!sample_buffer_idx];
			}

			__device__ __inline__ TracingPath& GetTracingPath(const vec2ui32& pixel)
			{
				return mp_tracing_paths[pixel.y * resolution.x + pixel.x];
			}


			// ray generation
		public:
			__device__ void GenerateSimpleRay(
				CudaSceneRay& ray,
				FullThread& thread,
				const CudaConstantKernel& ckernel)
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
					((0.5f / float(resolution.x)) * (ckernel.GetRNG().GetUnsignedUniform(thread) * 2.0f - 1.0f));
				ray.direction.y +=  // this --v-- should be x
					((0.5f / float(resolution.x)) * (ckernel.GetRNG().GetUnsignedUniform(thread) * 2.0f - 1.0f));

				// camera transformation
				CurrentCoordSystem().TransformBackward(ray.origin);
				CurrentCoordSystem().TransformBackward(ray.direction);
				ray.direction.Normalize();
				ray.origin += CurrentPosition();
			}
			__device__ void GenerateRay(
				CudaSceneRay& ray,
				FullThread& thread,
				const CudaConstantKernel& ckernel)
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
					((0.5f / float(resolution.x)) * (ckernel.GetRNG().GetUnsignedUniform(thread) * 2.0f - 1.0f));
				ray.direction.y +=  // this --v-- should be x
					((0.5f / float(resolution.x)) * (ckernel.GetRNG().GetUnsignedUniform(thread) * 2.0f - 1.0f));

				// focal point
				const vec3f focalPoint = ray.direction * focal_distance;

				// aperture distortion
				const float apertureAngle = ckernel.GetRNG().GetUnsignedUniform(thread) * CUDART_PI_F * 2.0f;
				const float apertureSample = sqrtf(ckernel.GetRNG().GetUnsignedUniform(thread)) * aperture;
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
			}

		
			// Spatio-temporal reprojection
		private:
			template <typename T>
			__device__ T Blend(const T& v1, const T& v2, const float& a)
			{
				return (v1 * a) + (v2 * (1.0f - a));
			}
		public:
			__device__ void Reproject(
				const vec2ui32& pixel)
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
}

#endif // !CUDA_CAMERA_CUH