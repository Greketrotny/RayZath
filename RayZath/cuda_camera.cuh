#ifndef CUDA_CAMERA_CUH
#define CUDA_CAMERA_CUH

#include "rzexception.h"
#include "camera.h"
#include "exist_flag.cuh"
#include "cuda_engine_parts.cuh"
#include "cuda_render_parts.cuh"

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
			float inv_passes_count;


			// sample image
			cudaArray* mp_array_sample_image;
			cudaSurfaceObject_t m_so_sample;
			// final image
			cudaArray* mp_array_final_image[2];
			cudaSurfaceObject_t m_so_final[2];
			// depth buffer
			cudaArray* mp_array_db[2];
			cudaSurfaceObject_t m_so_db[2];

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
			__host__ cudaArray* GetFinalImageArray(const uint32_t& idx) const
			{
				return mp_array_final_image[idx];
			}

		private:
			__host__ void CreateCudaSurface(
				const cudaChannelFormatDesc& cfd,
				cudaSurfaceObject_t& so,
				cudaArray*& array);
			__host__ void DestroyCudaSurface(
				cudaSurfaceObject_t& so,
				cudaArray*& array);
		public:
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
			__host__ __device__ float& GetInvPassesCount()
			{
				return inv_passes_count;
			}
			__device__ const float& GetAperture() const
			{
				return aperture;
			}
			__device__ const float& GetExposureTime() const
			{
				return exposure_time;
			}

			__device__ __inline__ void AppendSample(
				const Color<float>& sample,
				const uint32_t x, const uint32_t y)
			{
				float4 pixel;
				#if defined(__CUDACC__)
				surf2Dread<float4>(&pixel, m_so_sample, x * sizeof(pixel), y);
				pixel.x += sample.blue;
				pixel.y += sample.green;
				pixel.z += sample.red;
				pixel.w += sample.alpha;
				surf2Dwrite<float4>(pixel, m_so_sample, x * sizeof(pixel), y);
				#endif
			}
			__device__ __inline__ void SetSamplePixel(
				const Color<float>& sample,
				const uint32_t x, const uint32_t y)
			{
				float4 pixel;
				pixel.x = sample.blue;
				pixel.y = sample.green;
				pixel.z = sample.red;
				pixel.w = sample.alpha;
				#if defined(__CUDACC__)
				surf2Dwrite<float4>(pixel, m_so_sample, x * sizeof(pixel), y);
				#endif
			}
			__device__ __inline__ Color<float> GetSamplePixel(
				const uint32_t x, const uint32_t y)
			{
				float4 pixel;
				#if defined(__CUDACC__)
				surf2Dread<float4>(&pixel, m_so_sample, x * sizeof(pixel), y);
				#endif
				return Color<float>(pixel.z, pixel.y, pixel.x, pixel.w);
			}

			__device__ __inline__ void SetFinalPixel(
				const unsigned int buffer_idx,
				const Color<unsigned char>& color,
				const uint32_t x, const uint32_t y)
			{
				uchar4 pixel;
				pixel.x = color.blue;
				pixel.y = color.green;
				pixel.z = color.red;
				pixel.w = color.alpha;
				#if defined(__CUDACC__)
				surf2Dwrite<uchar4>(pixel, m_so_final[buffer_idx], x * sizeof(pixel), y);
				#endif
			}

			__device__ __inline__ void SetDepthBufferValue(
				const unsigned int buffer_idx,
				const float& value,
				const uint32_t x, const uint32_t y)
			{
				float1 depth{ value };
				#if defined(__CUDACC__)
				surf2Dwrite<float1>(depth, m_so_db[buffer_idx], x * sizeof(depth), y);
				#endif
			}
			__device__ __inline__ float GetDepthBufferValue(
				const unsigned int buffer_idx,
				const uint32_t x, const uint32_t y)
			{
				float1 depth;
				#if defined(__CUDACC__)
				surf2Dread<float1>(&depth, m_so_db[buffer_idx], x * sizeof(depth), y);
				#endif
				return depth.x;
			}

			__device__ __inline__ TracingPath& GetTracingPath(const uint32_t index)
			{
				return mp_tracing_paths[index];
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