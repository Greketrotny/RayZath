#include "cuda_engine_kernel.cuh"
#include "cuda_render_parts.cuh"

namespace RayZath
{
    namespace CudaKernel
    {
        __global__ void Kernel(CudaWorld* world, const int index)
        {
            CudaCamera* camera = &world->cameras[0];
            camera->position /= 2.0f;
            camera->position += cudaVec3<float>(1.0f, -3.0f, 0.0f);
            camera->rotation /= 4.0f;

            const uint64_t c_width = camera->width;
            const uint64_t c_height = camera->height;

            const uint64_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
            const uint64_t thread_x = thread_index % c_width;
            const uint64_t thread_y = thread_index / c_width;
            
            const float green = thread_x / static_cast<float>(c_width);
            const float blue = thread_y / static_cast<float>(c_height);
            camera->FinalImagePixel(index, thread_index) = CudaColor<unsigned char>(
                0x00, 
                green * green * 255.0f, 
                blue * blue * 255.0f);
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
