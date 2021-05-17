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
			using type = float4;
		};
		template<> struct CudaVectorType<uint16_t>
		{
			using type = ushort1;
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
		template<> __device__ __inline__ float4 CudaVectorTypeConvert(const vec3f& v)
		{
			return make_float4(v.x, v.y, v.z, 0.0f);
		}
		template<> __device__ __inline__ vec3f CudaVectorTypeConvert(const float4& v)
		{
			return vec3f(v.x, v.y, v.z);
		}
		template<> __device__ __inline__ uint16_t CudaVectorTypeConvert(const ushort1& v)
		{
			return v.x;
		}
		template<> __device__ __inline__ ushort1 CudaVectorTypeConvert(const uint16_t& v)
		{
			return make_ushort1(v);
		}


		template <typename T>
		struct CudaSurfaceBuffer
		{
		private:
			vec2ui32 m_resolution;
			cudaSurfaceObject_t m_so;
			cudaArray* mp_array;

		public:
			__host__ CudaSurfaceBuffer(
				const vec2ui32& resolution = vec2ui32(0u, 0u))
				: m_resolution(resolution)
				, m_so(0u)
				, mp_array(nullptr)
			{
				Allocate();
			}
			__host__ ~CudaSurfaceBuffer()
			{
				Deallocate();
			}

			__host__ void Reset(const vec2ui32& resolution)
			{
				Deallocate();
				m_resolution.x = std::max(resolution.x, 1u);
				m_resolution.y = std::max(resolution.y, 1u);
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
					m_resolution.x, m_resolution.y,
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
				const vec2ui32& point,
				const T& value)
			{
				#if defined(__CUDACC__)
				surf2Dwrite<CudaVectorType<T>::type>(
					CudaVectorTypeConvert<T, CudaVectorType<T>::type>(value),
					m_so, point.x * sizeof(CudaVectorType<T>::type), point.y);
				#endif
			}
			__device__ __inline__ T GetValue(
				const vec2ui32& point)
			{
				typename CudaVectorType<T>::type value;
				#if defined(__CUDACC__)
				surf2Dread<CudaVectorType<T>::type>(
					&value, m_so, point.x * sizeof(CudaVectorType<T>::type), point.y);
				#endif
				return CudaVectorTypeConvert<decltype(value), T>(value);
			}
			__device__ __inline__ void AppendValue(
				const vec2ui32& point,
				const T& value)
			{
				T v = GetValue(point);
				v += value;
				SetValue(point, v);
			}
		};


		template <typename T>
		struct CudaGlobalBuffer
		{
		private:
			vec2ui32 m_resolution;
			size_t m_pitch;
			T* mp_array;


		public:
			__host__ CudaGlobalBuffer(
			const vec2ui32& resolution = vec2ui32(0u, 0u))
				: m_resolution(resolution)
				, m_pitch(width)
				, mp_array(nullptr)
			{
				Allocate();
			}
			__host__ ~CudaGlobalBuffer()
			{
				Deallocate();
			}

			__host__ void Reset(const vec2ui32& resolution)
			{
				Deallocate();
				m_resolution.x = std::max(resolution.x, 1u);
				m_resolution.y = std::max(resolution.y, 1u);
				Allocate();
			}
			__host__ T* GetDataPtr()
			{
				return mp_array;
			}
		private:
			__host__ void Allocate()
			{
				if (mp_array != nullptr) return;
				CudaErrorCheck(cudaMallocPitch(
					&mp_array, 
					&m_pitch, 
					m_resolution.x * sizeof(T), m_resolution.y));
			}
			__host__ void Deallocate()
			{
				if (mp_array)
				{
					CudaErrorCheck(cudaFree(mp_array));
					mp_array = nullptr;
				}
			}

		public:
			__device__ __inline__ T& Value(
				const vec2ui32& point)
			{
				return *(((T*)((char*)mp_array + point.y * m_pitch)) + point.x);
			}
			__device__ __inline__ const T& Value(
				const vec2ui32& point) const
			{
				return *(((T*)((char*)mp_array + point.y * m_pitch)) + point.x);
			}
			__device__ __inline__ void SetValue(
				const T& value,
				const vec2ui32& point)
			{
				Value(point) = value;
			}
			__device__ __inline__ const T& GetValue(
				const vec2ui32& point) const
			{
				return Value(point);
			}
			__device__ __inline__ void AppendValue(
				const T& value,
				const vec2ui32& point)
			{
				Value(point) += value;
			}
		};
	}
}

#endif