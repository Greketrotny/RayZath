#include "cuda_engine_kernel.cuh"
#include "cuda_render_parts.cuh"

namespace RayZath
{
	namespace CudaKernel
	{
		__global__ void Kernel(
			CudaKernelData* const kernel_data,
			CudaWorld* const world, 
			const int index)
		{
/* set camera index */			CudaCamera* const camera = &world->cameras[0];
			if (!camera->Exist()) return;

			const size_t camera_width = camera->width;
			const size_t camera_height = camera->height;

			// calculate which pixel the thread correspond to
			const size_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_index >= camera_width * camera_height) return;


			if (thread_index == 0)
			{
				camera->position /= 2.0f;
				camera->rotation /= 4.0f;
				camera->rotation = cudaVec3<float>(1.0f, 2.0f, 3.0f);
			}
			__syncthreads();


			const size_t thread_x = thread_index % camera_width;
			const size_t thread_y = thread_index / camera_width;


			RayIntersection intersection;
			intersection.worldSpaceRay.direction = cudaVec3<float>(0.0f, 0.0f, 1.0f);

			// ray to screen deflection
			float xShift = __tanf(camera->fov * 0.5f);
			float yShift = -__tanf(camera->fov * 0.5f) / camera->aspect_ratio;
			intersection.worldSpaceRay.direction.x = ((thread_x / (float)camera_width - 0.5f) * xShift);
			intersection.worldSpaceRay.direction.y = ((thread_y / (float)camera_height - 0.5f) * yShift);

			// pixel position distortion (antialiasing)
		//	intersection.worldSpaceRay.direction.x += ((0.5f / (float)cameraWidth) * renderingKernel->randomNumbers.GetSignedUniform());
		//	intersection.worldSpaceRay.direction.y += ((0.5f / (float)cameraHeight) * renderingKernel->randomNumbers.GetSignedUniform());

			// focal point
			cudaVec3<float> focalPoint = intersection.worldSpaceRay.direction * camera->focal_distance;

			// aperture distortion
		//	float apertureAngle = renderingKernel->randomNumbers.GetUnsignedUniform() * 6.28318530f;
		//	float apertureSample = renderingKernel->randomNumbers.GetUnsignedUniform() * camera->aperture;
		//	intersection.worldSpaceRay.origin += cudaVec3<float>(
		//		apertureSample * __sinf(apertureAngle),
		//		apertureSample * __cosf(apertureAngle),
		//		0.0f);

			// depth of field ray
			intersection.worldSpaceRay.direction = focalPoint - intersection.worldSpaceRay.origin;


			// ray direction rotation
			intersection.worldSpaceRay.direction.RotateZ(camera->rotation.z);
			intersection.worldSpaceRay.direction.RotateX(camera->rotation.x);
			intersection.worldSpaceRay.direction.RotateY(camera->rotation.y);
			intersection.worldSpaceRay.direction.Normalize();

			// ray origin rotation
			intersection.worldSpaceRay.origin.RotateZ(camera->rotation.z);
			intersection.worldSpaceRay.origin.RotateX(camera->rotation.x);
			intersection.worldSpaceRay.origin.RotateY(camera->rotation.y);

			// ray transposition
			intersection.worldSpaceRay.origin += camera->position;


			// trace ray from camera
		//	TracingPath* tracingPath = &camera->GetTracingPath(threadIndex);
		//	tracingPath->ResetPath();

			//camera->SamplingImagePixel(threadIndex) += CudaColor<float>(0.0f, 1.0f, 0.0f);
			//return;

		//	TraceRay(*world, *renderingKernel, *tracingPath, intersection);
		//	camera->SamplingImagePixel(threadIndex) += tracingPath->CalculateFinalColor();

			const float green = thread_x / static_cast<float>(camera_width);
			const float blue = thread_y / static_cast<float>(camera_height);
			float value = kernel_data->randomNumbers.GetUnsignedUniform();
			value = ((value > 0.5f) ? 255.0f : 0.0f);

			const float red = (world->spheres[0].position.x == -2.0f) ? 255.0f : 0.0f;
			/*camera->FinalImagePixel(index, thread_index) = CudaColor<unsigned char>(
				green * green * 255.0f,
				kernel_data->randomNumbers.GetUnsignedUniform() * 255.0f,
				blue * blue * 255.0f);*/
			camera->FinalImagePixel(index, thread_index) = CudaColor<unsigned char>(
				red,
				value,
				value);

		}

		void CallKernel()
		{
			return;

			const size_t test_size = 1024;
			cudaVec3<float>* h_vec = new cudaVec3<float>[test_size];
			for (size_t i = 0; i < test_size; i++)
				h_vec[i] = cudaVec3<float>(
					(rand() % RAND_MAX) / static_cast<float>(RAND_MAX), i / 2.0f, i * 1.5f);

			cudaVec3<float>* d_vec = nullptr;
			CudaErrorCheck(cudaMalloc((void**)&d_vec, test_size * sizeof(cudaVec3<float>)));

			CudaErrorCheck(cudaMemcpy(d_vec, h_vec, test_size * sizeof(cudaVec3<float>), cudaMemcpyHostToDevice));

		   // Kernel << <1u, 1u >> > (d_vec, test_size);

			CudaErrorCheck(cudaDeviceSynchronize());
			CudaErrorCheck(cudaGetLastError());

			cudaVec3<float>* h_vec_result = new cudaVec3<float>[test_size];
			CudaErrorCheck(cudaMemcpy(
				h_vec_result, d_vec,
				test_size * sizeof(cudaVec3<float>),
				cudaMemcpyDeviceToHost));


			for (size_t i = 0; i < test_size; i++)
			{
				h_vec[i] /= 2.0f;
				h_vec[i] *= 1.5f;

				if (h_vec[i].x != h_vec_result[i].x ||
					h_vec[i].y != h_vec_result[i].y ||
					h_vec[i].z != h_vec_result[i].z)
					throw Exception(__FILE__, __LINE__, L"kernel test case does not match!");
			}

			CudaErrorCheck(cudaFree(d_vec));
			delete[] h_vec;
			delete[] h_vec_result;
		}
	}
}
