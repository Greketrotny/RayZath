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

		CudaColor<float>* sampling_image;
		unsigned int samples_count;
		CudaColor<unsigned char>* final_image[2];
		//CudaEngineKernel::TracingPath* tracingPaths;

		static HostPinnedMemory hostPinnedMemory;


	public:
		__host__ CudaCamera();
		__host__ ~CudaCamera();


		__host__ void Reconstruct(
			const Camera& hCamera,
			cudaStream_t& mirror_stream);
	public:
		/*__device__ __inline__ CudaColor<float>& SamplingImagePixel(unsigned int x, unsigned int y)
		{
			return samplingImage[y * width + x];
		}*/
		/*__device__ __inline__ CudaColor<float>& SamplingImagePixel(uint64_t index)
		{
			return sampling_image[index];
		}*/
		/*__device__ __inline__ CudaColor<unsigned char>& FinalImagePixel(unsigned int bufferIndex, unsigned int x, unsigned int y)
		{
			return finalImage[bufferIndex][y * width + x];
		}*/
		__device__ __inline__ CudaColor<unsigned char>& FinalImagePixel(unsigned int bufferIndex, uint64_t index)
		{
			return final_image[bufferIndex][index];
		}
		/*__device__ __inline__ CudaEngineKernel::TracingPath& GetTracingPath(unsigned int threadX, unsigned int threadY)
		{
			return tracingPaths[threadY * width + threadX];
		}*/
		/*__device__ __inline__ CudaEngineKernel::TracingPath& GetTracingPath(uint64_t index)
		{
			return tracingPaths[index];
		}*/
		//__host__ CudaColor<unsigned char>* FinalImageGetAddress(unsigned int bufferIndex);
	};
}

#endif // !CUDA_CAMERA_CUH