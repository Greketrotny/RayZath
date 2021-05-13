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
			vec3f position;
			vec3f rotation;
			CudaCoordSystem coord_system;

			uint32_t width, height;
			float aspect_ratio;
			bool enabled;
			float fov;

			float focal_distance;
			float aperture;
			float exposure_time;

			uint32_t passes_count;
			bool sample_buffer_idx;

			CudaSurfaceBuffer<ColorF> m_sample_image_buffer[2];
			CudaGlobalBuffer<float> m_sample_depth_buffer;
			CudaSurfaceBuffer<ColorU> m_final_image_buffer[2];
			CudaSurfaceBuffer<float> m_final_depth_buffer[2];

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
			__host__ __device__ const vec3f& GetPosition() const
			{
				return position;
			}
			__host__ __device__ const uint32_t& GetWidth() const
			{
				return width;
			}
			__host__ __device__ const uint32_t& GetHeight() const
			{
				return height;
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

			__device__ __inline__ CudaGlobalBuffer<float>& SampleDepthBuffer()
			{
				return m_sample_depth_buffer;
			}
			__host__ __device__ __inline__ CudaSurfaceBuffer<ColorU>& FinalImageBuffer(const uint32_t& idx)
			{
				return m_final_image_buffer[idx];
			}
			__host__ __device__ __inline__ CudaSurfaceBuffer<float>& FinalDepthBuffer(const uint32_t& idx)
			{
				return m_final_depth_buffer[idx];
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

			__device__ __inline__ TracingPath& GetTracingPath(const uint32_t idx)
			{
				return mp_tracing_paths[idx];
			}

			__device__ void Reproject(
				const uint32_t& x, const uint32_t& y)
			{
				vec3f p = SpaceBuffer().GetValue(x, y);
				p -= position;
				coord_system.TransformForward(p);

				if (p.z < 0.0f)
					return;

				p /= p.z;
				const float x_shift = cui_tanf(fov * 0.5f);
				const float y_shift = -x_shift / aspect_ratio;

				const float screen_x = ((p.x / x_shift) + 0.5f) * width + 0.5f;
				const float screen_y = ((p.y / y_shift) + 0.5f) * height + 0.5f;
				if (screen_x >= 0.0f && screen_x < width && screen_y >= 0.0f && screen_y < height)
				{
					const float d = vec3f::Distance(position, p);
					if (d < SampleDepthBuffer().GetValue(screen_x, screen_y))
					{
						// reprojection
						EmptyImageBuffer().SetValue(
							SampleImageBuffer().GetValue(x, y), screen_x, screen_y);
						EmptyPassesBuffer().SetValue(
							PassesBuffer().GetValue(x, y), screen_x, screen_y);
						SampleDepthBuffer().SetValue(d, screen_x, screen_y);
					}
				}
			}

			// ray generation
		public:
			__device__ __inline__ void GenerateRay(
				CudaSceneRay& ray,
				ThreadData& thread,
				const CudaConstantKernel& ckernel)
			{
				ray.direction = vec3f(0.0f, 0.0f, 1.0f);

				// ray to screen deflection
				const float x_shift = cui_tanf(fov * 0.5f);
				const float y_shift = -x_shift / aspect_ratio;
				ray.direction.x = ((thread.thread_x / (float)width - 0.5f) * x_shift);
				ray.direction.y = ((thread.thread_y / (float)height - 0.5f) * y_shift);

				// pixel position distortion (antialiasing)
				ray.direction.x +=
					((0.5f / (float)width) * (ckernel.GetRNG().GetUnsignedUniform(thread) * 2.0f - 1.0f));
				ray.direction.y +=
					((0.5f / (float)height) * (ckernel.GetRNG().GetUnsignedUniform(thread) * 2.0f - 1.0f));

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
				coord_system.TransformBackward(ray.origin);
				coord_system.TransformBackward(ray.direction);
				ray.direction.Normalize();
				ray.origin += position;
			}
		};
	}
}

#endif // !CUDA_CAMERA_CUH