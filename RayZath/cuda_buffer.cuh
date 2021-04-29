#ifndef CUDA_BUFFER_CUH
#define CUDA_BUFFER_CUH

#include "cuda_render_parts.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		template <typename T>
		struct CudaVectorType
		{
			using type = T;
		};
		template<> struct CudaVectorType<ColorF>
		{
			using type = float4;
		};
		template<> struct CudaVectorType<ColorU>
		{
			using type = uchar4;
		};
		template<> struct CudaVectorType<vec3f>
		{
			using type = float3;
		};

		template<typename T1, typename T2>
		__device__ __inline__ T2 CudaVectorTypeConvert(const T1& t1)
		{
			return T2(t1);
		}
		template<> __device__ __inline__ float4 CudaVectorTypeConvert(const ColorF& c)
		{
			return float4{ c.red, c.green, c.blue, c.alpha };
		}
		template<> __device__ __inline__ ColorF CudaVectorTypeConvert(const float4& c)
		{
			return ColorF(c.x, c.y, c.z, c.w);
		}
		template<> __device__ __inline__ uchar4 CudaVectorTypeConvert(const ColorU& c)
		{
			return uchar4{ c.red, c.green, c.blue, c.alpha };
		}
		template<> __device__ __inline__ ColorU CudaVectorTypeConvert(const uchar4& c)
		{
			return ColorU(c.x, c.y, c.z, c.w);
		}
		template<> __device__ __inline__ float3 CudaVectorTypeConvert(const vec3f& v)
		{
			return float3{ v.x, v.y, v.z };
		}
		template<> __device__ __inline__ vec3f CudaVectorTypeConvert(const float3& v)
		{
			return vec3f(v.x, v.y, v.z);
		}


		template <typename T>
		struct CudaSurfaceBuffer
		{
		private:
			uint32_t m_width, m_height;
			cudaSurfaceObject_t m_so;
			cudaArray* mp_array;

		public:
			__host__ CudaSurfaceBuffer(const uint32_t& width = 0u, const uint32_t& height = 0u)
				: m_width(std::max(width, 1u))
				, m_height(std::max(height, 1u))
				, m_so(0u)
				, mp_array(nullptr)
			{
				Allocate();
			}
			__host__ ~CudaSurfaceBuffer()
			{
				Deallocate();
			}

			__host__ void Reset(const uint32_t& width, const uint32_t& height)
			{
				Deallocate();
				m_width = std::max(width, 1u);
				m_height = std::max(height, 1u);
				Allocate();
			}
			__host__ cudaArray* GetCudaArray()
			{
				return mp_array;
			}
		private:
			__host__ void Allocate()
			{
				if (m_so != 0u || mp_array != nullptr) return;

				// allocate cuda array
				auto cd = cudaCreateChannelDesc<CudaVectorType<T>::type>();
				CudaErrorCheck(cudaMallocArray(
					&mp_array,
					&cd,
					m_width, m_height,
					cudaArraySurfaceLoadStore));

				// create resource description
				cudaResourceDesc rd;
				std::memset(&rd, 0, sizeof(rd));
				rd.resType = cudaResourceTypeArray;
				rd.res.array.array = mp_array;
				m_so = 0u;
				CudaErrorCheck(cudaCreateSurfaceObject(&m_so, &rd));
			}
			__host__ void Deallocate()
			{
				if (m_so)
				{
					CudaErrorCheck(cudaDestroySurfaceObject(m_so));
					m_so = 0u;
				}
				if (mp_array)
				{
					CudaErrorCheck(cudaFreeArray(mp_array));
					mp_array = nullptr;
				}
			}

		public:
			__device__ __inline__ void SetValue(
				const T& value,
				const uint32_t& x, const uint32_t& y)
			{
				#if defined(__CUDACC__)
				surf2Dwrite<CudaVectorType<T>::type>(
					CudaVectorTypeConvert<T, CudaVectorType<T>::type>(value),
					m_so, x * sizeof(CudaVectorType<T>::type), y);
				#endif
			}
			__device__ __inline__ T GetValue(
				const uint32_t& x, const uint32_t& y)
			{
				typename CudaVectorType<T>::type value;
				#if defined(__CUDACC__)
				surf2Dread<CudaVectorType<T>::type>(
					&value, m_so, x * sizeof(CudaVectorType<T>::type), y);
				#endif
				return CudaVectorTypeConvert<decltype(value), T>(value);
			}
			__device__ __inline__ void AppendValue(
				const T& value,
				const uint32_t& x, const uint32_t& y)
			{
				T v = GetValue(x, y);
				v += value;
				SetValue(v, x, y);
			}
		};
	}
}

#endif