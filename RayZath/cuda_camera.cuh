#ifndef CUDA_CAMERA_CUH
#define CUDA_CAMERA_CUH

#include "rzexception.h"
#include "camera.h"
#include "exist_flag.cuh"
#include "cuda_engine_parts.cuh"
#include "cuda_render_parts.cuh"

namespace RayZath
{
	class CudaCamera : public WithExistFlag
	{
	public:
		cudaVec3<float> position;
		cudaVec3<float> rotation;

		uint64_t width, height;
		float aspect_ratio;
		bool enabled;
		float fov;

		float focal_distance;
		float aperture;

		unsigned int samples_count;
	public:
		cudaArray* mp_sample_image_array;
		cudaSurfaceObject_t m_so_sample;
		cudaArray* mp_final_image_array[2];
		cudaSurfaceObject_t m_so_final[2];

		TracingPath* mp_tracing_paths;
	public:
		static HostPinnedMemory hostPinnedMemory;


	public:
		__host__ CudaCamera();
		__host__ ~CudaCamera();


		__host__ void Reconstruct(
			Camera& hCamera,
			cudaStream_t& mirror_stream);
	public:
		__device__ __inline__ void AppendSample(
			const CudaColor<float>& sample, 
			uint64_t x, uint64_t y)
		{
			float4 pixel;
			#if defined(__CUDACC__)
			surf2Dread<float4>(&pixel, m_so_sample, x * sizeof(pixel), y);
			pixel.x += sample.red;
			pixel.y += sample.green;
			pixel.z += sample.blue;
			surf2Dwrite<float4>(pixel, m_so_sample, x * sizeof(pixel), y);
			#endif
		}
		__device__ __inline__ void SetSample(
			const CudaColor<float>& sample, 
			uint64_t x, uint64_t y)
		{
			float4 pixel;
			pixel.x = sample.red;
			pixel.y = sample.green;
			pixel.z = sample.blue;
			pixel.w = 0u;
			#if defined(__CUDACC__)
			surf2Dwrite<float4>(pixel, m_so_sample, x * sizeof(pixel), y);
			#endif
		}
		__device__ __inline__ CudaColor<float> GetSample(
			uint64_t x, uint64_t y)
		{
			float4 pixel;
			#if defined(__CUDACC__)
			surf2Dread<float4>(&pixel, m_so_sample, x * sizeof(pixel), y);
			#endif
			return CudaColor<float>(pixel.x, pixel.y, pixel.z);
		}

		__device__ __inline__ void SetFinalPixel(
			const unsigned int buffer, 
			const CudaColor<unsigned char>& color, 
			uint64_t x, uint64_t y)
		{
			uchar4 pixel;
			pixel.x = color.red;
			pixel.y = color.green;
			pixel.z = color.blue;
			pixel.w = color.alpha;
			#if defined(__CUDACC__)
			surf2Dwrite<uchar4>(pixel, m_so_final[buffer], x * sizeof(pixel), y);
			#endif
		}

		__device__ __inline__ TracingPath& GetTracingPath(size_t index)
		{
			return mp_tracing_paths[index];
		}
	};
}

#endif // !CUDA_CAMERA_CUH