#include "cuda_camera.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		HostPinnedMemory CudaCamera::hostPinnedMemory(0x10000u);

		__host__ CudaCamera::CudaCamera()
			: width(0), height(0)
			, aspect_ratio(1.0f)
			, enabled(true)
			, fov(2.0f)
			, focal_distance(10.0f)
			, aperture(0.01f)
			, exposure_time(1.0f / 60.0f)
			, passes_count(0u)
			, inv_passes_count(1.0f)
			, mp_array_sample_image(nullptr)
			, m_so_sample(0u)
			, mp_array_final_image{ nullptr, nullptr }
			, m_so_final{ 0u, 0u }
			, mp_array_db{ nullptr, nullptr }
			, m_so_db{ 0u, 0u }
			, mp_tracing_paths(nullptr)
		{}
		__host__ CudaCamera::~CudaCamera()
		{
			// destroy sample buffers
			DestroyCudaSurface(m_so_sample, mp_array_sample_image);

			// destroy final image buffers
			DestroyCudaSurface(m_so_final[0], mp_array_final_image[0]);
			DestroyCudaSurface(m_so_final[1], mp_array_final_image[1]);

			// destroy depth buffer
			DestroyCudaSurface(m_so_db[0], mp_array_db[0]);
			DestroyCudaSurface(m_so_db[1], mp_array_db[1]);

			// destroy tracing paths
			if (mp_tracing_paths) CudaErrorCheck(cudaFree(mp_tracing_paths));
			mp_tracing_paths = nullptr;
		}

		__host__ void CudaCamera::Reconstruct(
			const CudaWorld& hCudaWorld, 
			const Handle<Camera>& hCamera, 
			cudaStream_t& mirror_stream)
		{
			if (!hCamera->GetStateRegister().IsModified()) return;

			position = hCamera->GetPosition();
			rotation = hCamera->GetRotation();

			coord_system = hCamera->GetCoordSystem();

			aspect_ratio = hCamera->GetAspectRatio();
			fov = hCamera->GetFov().value();
			focal_distance = hCamera->GetFocalDistance();
			aperture = hCamera->GetAperture();
			exposure_time = hCamera->GetExposureTime();
			enabled = hCamera->Enabled();

			if (width != hCamera->GetWidth() || height != hCamera->GetHeight())
			{// resize buffers to match size of hostCamera resolution

				// [>] Release CudaCamera resources
				// destroy sample buffers
				DestroyCudaSurface(m_so_sample, mp_array_sample_image);

				// destroy final image buffers
				DestroyCudaSurface(m_so_final[0], mp_array_final_image[0]);
				DestroyCudaSurface(m_so_final[1], mp_array_final_image[1]);

				// destroy depth buffer
				DestroyCudaSurface(m_so_db[0], mp_array_db[0]);
				DestroyCudaSurface(m_so_db[1], mp_array_db[1]);

				// destroy tracing paths
				if (mp_tracing_paths) CudaErrorCheck(cudaFree(mp_tracing_paths));


				// [>] Update CudaCamera resolution
				width = hCamera->GetWidth();
				height = hCamera->GetHeight();


				// [>] Reallocate resources
				// create sample buffer
				CreateCudaSurface(
					cudaCreateChannelDesc<float4>(), 
					m_so_sample, mp_array_sample_image);

				// create final image surfaces
				CreateCudaSurface(
					cudaCreateChannelDesc<uchar4>(),
					m_so_final[0], mp_array_final_image[0]);
				CreateCudaSurface(
					cudaCreateChannelDesc<uchar4>(),
					m_so_final[1], mp_array_final_image[1]);

				// create depth buffer
				CreateCudaSurface(
					cudaCreateChannelDesc<float1>(),
					m_so_db[0], mp_array_db[0]);
				CreateCudaSurface(
					cudaCreateChannelDesc<float1>(),
					m_so_db[1], mp_array_db[1]);				

				// allocate memory for tracing paths
				CudaErrorCheck(cudaMalloc(
					(void**)&mp_tracing_paths, 
					size_t(width) * size_t(height) * size_t(sizeof(*mp_tracing_paths))));


				// [>] Resize hostPinnedMemory for mirroring
				this->hostPinnedMemory.SetMemorySize(
					std::min(
						width * height * uint32_t(sizeof(Color<unsigned char>)),
						0x100000u)); // max 1MiB
			}

			hCamera->GetStateRegister().MakeUnmodified();
		}

		__host__ void CudaCamera::CreateCudaSurface(
			const cudaChannelFormatDesc& cfd,
			cudaSurfaceObject_t& so,
			cudaArray*& array)
		{
			// allocate array
			CudaErrorCheck(cudaMallocArray(
				&array,
				&cfd,
				width, height,
				cudaArraySurfaceLoadStore));

			// create resource description
			cudaResourceDesc rd;
			std::memset(&rd, 0, sizeof(rd));
			rd.resType = cudaResourceTypeArray;
			rd.res.array.array = array;
			so = 0u;
			CudaErrorCheck(cudaCreateSurfaceObject(&so, &rd));
		}
		__host__ void CudaCamera::DestroyCudaSurface(
			cudaSurfaceObject_t& so,
			cudaArray*& array)
		{
			if (so)
			{
				CudaErrorCheck(cudaDestroySurfaceObject(so));
				so = 0u;
			}
			if (array)
			{
				CudaErrorCheck(cudaFreeArray(array));
				array = nullptr;
			}
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	}
}