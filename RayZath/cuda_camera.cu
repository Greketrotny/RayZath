#include "cuda_camera.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		HostPinnedMemory CudaCamera::hostPinnedMemory(0xFFFF);

		__host__ CudaCamera::CudaCamera()
			: width(0), height(0)
			, aspect_ratio(1.0f)
			, enabled(true)
			, fov(2.0f)
			, focal_distance(10.0f)
			, aperture(0.01f)
			, passes_count(0u)
			, mp_sample_image_array(nullptr)
			, m_so_sample(0u)
			, mp_final_image_array{ 0u, 0u }
			, m_so_final{ 0u, 0u }
			, mp_tracing_paths(nullptr)
		{}
		__host__ CudaCamera::~CudaCamera()
		{
			// destroy sample image surface
			if (m_so_sample) CudaErrorCheck(cudaDestroySurfaceObject(m_so_sample));
			m_so_sample = 0u;
			if (mp_sample_image_array) CudaErrorCheck(cudaFreeArray(mp_sample_image_array));
			this->mp_sample_image_array = nullptr;

			// destroy final image surfaces
			for (uint32_t i = 0u; i < 2u; i++)
			{
				if (m_so_final[i]) CudaErrorCheck(cudaDestroySurfaceObject(m_so_final[i]));
				m_so_final[i] = 0;
				if (mp_final_image_array[i]) CudaErrorCheck(cudaFreeArray(mp_final_image_array[i]));
				this->mp_final_image_array[i] = nullptr;
			}

			// destroy tracing paths
			if (mp_tracing_paths) CudaErrorCheck(cudaFree(mp_tracing_paths));
			mp_tracing_paths = nullptr;
		}

		__host__ void CudaCamera::Reconstruct(Camera& hCamera, cudaStream_t& mirror_stream)
		{
			if (!hCamera.GetStateRegister().IsModified()) return;

			position = hCamera.GetPosition();
			rotation = hCamera.GetRotation();

			aspect_ratio = hCamera.GetAspectRatio();
			fov = hCamera.GetFov().value();
			focal_distance = hCamera.GetFocalDistance();
			aperture = hCamera.GetAperture();
			enabled = hCamera.Enabled();

			if (width != hCamera.GetWidth() || height != hCamera.GetHeight())
			{// resize pixel map to match size with hostCamera resolution

				// [>] Release CudaCamera resources
				// destroy surface objects
				if (m_so_sample) CudaErrorCheck(cudaDestroySurfaceObject(m_so_sample));
				if (m_so_final[0]) CudaErrorCheck(cudaDestroySurfaceObject(m_so_final[0]));
				if (m_so_final[1]) CudaErrorCheck(cudaDestroySurfaceObject(m_so_final[1]));

				// free sampling image, final image and tracing paths memory
				if (mp_sample_image_array)		CudaErrorCheck(cudaFreeArray(mp_sample_image_array));
				if (mp_final_image_array[0])	CudaErrorCheck(cudaFreeArray(mp_final_image_array[0]));
				if (mp_final_image_array[1])	CudaErrorCheck(cudaFreeArray(mp_final_image_array[1]));
				if (mp_tracing_paths)			CudaErrorCheck(cudaFree(mp_tracing_paths));


				// [>] Update CudaCamera resolution
				width = hCamera.GetWidth();
				height = hCamera.GetHeight();


				// [>] Reallocate resources
				// create sample image surface
				cudaChannelFormatDesc cd_sample = cudaCreateChannelDesc<float4>();
				CudaErrorCheck(cudaMallocArray(
					&mp_sample_image_array, 
					&cd_sample, 
					width, height,
					cudaArraySurfaceLoadStore));

				cudaResourceDesc rd_sample;
				memset(&rd_sample, 0, sizeof(rd_sample));
				rd_sample.resType = cudaResourceTypeArray;

				rd_sample.res.array.array = mp_sample_image_array;
				m_so_sample = 0;
				CudaErrorCheck(cudaCreateSurfaceObject(&m_so_sample, &rd_sample));

				// create final image surfaces
				cudaChannelFormatDesc cd_final = cudaCreateChannelDesc<uchar4>();
				CudaErrorCheck(cudaMallocArray(
					&mp_final_image_array[0], 
					&cd_final, 
					width, height, 
					cudaArraySurfaceLoadStore));
				CudaErrorCheck(cudaMallocArray(
					&mp_final_image_array[1], 
					&cd_final, 
					width, height, 
					cudaArraySurfaceLoadStore));

				cudaResourceDesc rd_final;
				memset(&rd_final, 0, sizeof(rd_final));
				rd_final.resType = cudaResourceTypeArray;

				rd_final.res.array.array = mp_final_image_array[0];
				m_so_final[0] = 0;
				CudaErrorCheck(cudaCreateSurfaceObject(&m_so_final[0], &rd_final));
				rd_final.res.array.array = mp_final_image_array[1];
				m_so_final[1] = 0;
				CudaErrorCheck(cudaCreateSurfaceObject(&m_so_final[1], &rd_final));

				// allocate memory for tracing paths
				CudaErrorCheck(cudaMalloc(
					(void**)&mp_tracing_paths, 
					width * height * sizeof(*mp_tracing_paths)));


				// [>] Resize hostPinnedMemory for mirroring
				this->hostPinnedMemory.SetMemorySize(
					std::min(
						width * height * sizeof(CudaColor<unsigned char>),
						0x100000ull)); // max 1MB
				passes_count = 0u;
			}

			hCamera.GetStateRegister().MakeUnmodified();
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	}
}