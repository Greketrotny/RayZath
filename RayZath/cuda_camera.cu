#include "cuda_camera.cuh"

namespace RayZath
{
	HostPinnedMemory CudaCamera::hostPinnedMemory(0xFFFF);

	__host__ CudaCamera::CudaCamera()
		: width(0), height(0)
		, max_width(0), max_height(0)
		, aspect_ratio(1.0f)
		, enabled(true)
		, fov(2.0f)
		, focal_distance(10.0f)
		, aperture(0.01f)
		, sampling_image(nullptr)
		, samples_count(0)
		//, tracingPaths(nullptr)
	{
		final_image[0] = nullptr;
		final_image[1] = nullptr;
	}
	__host__ CudaCamera::~CudaCamera()
	{
		if (sampling_image) CudaErrorCheck(cudaFree(sampling_image));
		this->sampling_image = nullptr;

		if (final_image[0]) CudaErrorCheck(cudaFree(final_image[0]));
		final_image[0] = nullptr;
		if (final_image[1]) CudaErrorCheck(cudaFree(final_image[1]));
		final_image[1] = nullptr;

		//if (this->tracingPaths) CudaErrorCheck(cudaFree(this->tracingPaths));
		//this->tracingPaths = nullptr;

		max_width = 0u;
		max_height = 0u;
	}

	__host__ void CudaCamera::Reconstruct(const Camera& hCamera, cudaStream_t* mirror_stream)
	{
		position = hCamera.GetPosition();
		rotation = hCamera.GetRotation();

		width = hCamera.GetWidth();
		height = hCamera.GetHeight();

		if (max_width * max_height != hCamera.GetMaxWidth() * hCamera.GetMaxHeight())
		{// resize pixel map to match size with hostCamera resolution

			// free sampling image and final image memory
			if (sampling_image)		CudaErrorCheck(cudaFree(sampling_image));
			if (final_image[0])		CudaErrorCheck(cudaFree(final_image[0]));
			if (final_image[1])		CudaErrorCheck(cudaFree(final_image[1]));
			//if (this->tracingPaths)		CudaErrorCheck(cudaFree(this->tracingPaths));

			// update max width and max height
			max_width = hCamera.GetMaxWidth();
			max_height = hCamera.GetMaxHeight();

			// allocate device memory for sampingImage and final_image
			CudaErrorCheck(cudaMalloc((void**)&sampling_image, max_width * max_height * sizeof(*sampling_image)));
			CudaErrorCheck(cudaMalloc((void**)&final_image[0], max_width * max_height * sizeof(*(final_image[0]))));
			CudaErrorCheck(cudaMalloc((void**)&final_image[1], max_width * max_height * sizeof(*(final_image[1]))));
			//CudaErrorCheck(cudaMalloc((void**)&this->tracingPaths, max_width * max_height * sizeof(*tracingPaths)));

			// resize hostPinnedMemory for mirroring
			this->hostPinnedMemory.SetMemorySize(std::min(max_width * max_height * sizeof(*sampling_image), uint64_t(0xFFFFFFllu)));
			samples_count = 0;
		}
	}
	/*__host__ CudaColor<unsigned char>* CudaCamera::FinalImageGetAddress(unsigned int bufferIndex)
	{
		return final_image[bufferIndex];
	}*/
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}