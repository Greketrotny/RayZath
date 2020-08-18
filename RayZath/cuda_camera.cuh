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
		uint64_t max_width, max_height;
		float aspect_ratio;
		bool enabled;
		float fov;

		float focal_distance;
		float aperture;

		unsigned int samples_count;
	private:
		CudaColor<float>* sampling_image;
		CudaColor<unsigned char>* final_image[2];
		TracingPath* mp_tracing_paths;
	public:
		static HostPinnedMemory hostPinnedMemory;


	public:
		__host__ CudaCamera();
		__host__ ~CudaCamera();


		__host__ void Reconstruct(
			Camera& hCamera,
			cudaStream_t& mirror_stream);
		__host__ CudaColor<unsigned char>* GetFinalImageAddress(const unsigned int buffer_index);
	public:
		__device__ __inline__ CudaColor<float>& SamplingImagePixel(uint64_t index)
		{
			return sampling_image[index];
		}
		__device__ __inline__ CudaColor<unsigned char>& FinalImagePixel(
			const unsigned int buffer_index, 
			uint64_t index)
		{
			return final_image[buffer_index][index];
		}
		__device__ __inline__ TracingPath& GetTracingPath(size_t index)
		{
			return mp_tracing_paths[index];
		}
	};
}

#endif // !CUDA_CAMERA_CUH