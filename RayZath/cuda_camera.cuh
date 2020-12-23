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
		public:
			cudaVec3<float> position;
			cudaVec3<float> rotation;

			uint32_t width, height;
			float aspect_ratio;
			bool enabled;
			float fov;

			float focal_distance;
			float aperture;

			uint32_t passes_count;
		public:
			// sample image
			cudaArray* mp_sample_image_array;
			cudaSurfaceObject_t m_so_sample;
			// final image
			cudaArray* mp_final_image_array[2];
			cudaSurfaceObject_t m_so_final[2];

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
			__device__ __inline__ void AppendSample(
				const CudaColor<float>& sample,
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
			__device__ __inline__ void SetSample(
				const CudaColor<float>& sample,
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
			__device__ __inline__ CudaColor<float> GetSample(
				const uint32_t x, const uint32_t y)
			{
				float4 pixel;
				#if defined(__CUDACC__)
				surf2Dread<float4>(&pixel, m_so_sample, x * sizeof(pixel), y);
				#endif
				return CudaColor<float>(pixel.z, pixel.y, pixel.x, pixel.w);
			}

			__device__ __inline__ void SetFinalPixel(
				const unsigned int buffer,
				const CudaColor<unsigned char>& color,
				const uint32_t x, const uint32_t y)
			{
				uchar4 pixel;
				pixel.x = color.blue;
				pixel.y = color.green;
				pixel.z = color.red;
				pixel.w = color.alpha;
				#if defined(__CUDACC__)
				surf2Dwrite<uchar4>(pixel, m_so_final[buffer], x * sizeof(pixel), y);
				#endif
			}

			__device__ __inline__ TracingPath& GetTracingPath(const uint32_t index)
			{
				return mp_tracing_paths[index];
			}
		};
	}
}

#endif // !CUDA_CAMERA_CUH