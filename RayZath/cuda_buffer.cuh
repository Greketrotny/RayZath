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

		
		template <typename T, typename... Types>
		constexpr bool is_any_of_v = std::disjunction_v<std::is_same<T, Types>...>;
		template <typename T>
		constexpr bool is_integral_v = is_any_of_v<T,
			ColorU, vec2ui16, vec2ui32>;

		template <typename T, bool F>
		struct ReadType;
		template <> struct ReadType<ColorU, true> { using type = ColorF; };
		template <> struct ReadType<ColorU, false> { using type = ColorU; };
		template <> struct ReadType<float, true> { using type = float; };
		template <> struct ReadType<float, false> { using type = float; };



		template <typename T>
		struct CudaSurfaceBuffer
		{
		private:
			vec2ui32 m_resolution;
			cudaSurfaceObject_t m_so;
			cudaArray* mp_array;

		public:
			__host__ CudaSurfaceBuffer(
				const vec2ui32& resolution = vec2ui32(1u, 1u))
				: m_resolution(resolution)
				, m_so(0u)
				, mp_array(nullptr)
			{
				if (m_resolution.x == 0u) m_resolution.x = 1u;
				if (m_resolution.y == 0u) m_resolution.y = 1u;
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


		template <typename T, bool normalized_read = true>
		struct CudaTextureBuffer
			: public WithExistFlag
		{
		private:
			cudaResourceDesc m_res_desc;
			cudaTextureDesc m_texture_desc;
			cudaArray* mp_texture_array;
			cudaTextureObject_t m_texture_object;


		public:
			__host__ CudaTextureBuffer()
				: mp_texture_array(nullptr)
				, m_texture_object(0ull)
			{}
			__host__ ~CudaTextureBuffer()
			{
				if (m_texture_object) CudaErrorCheck(cudaDestroyTextureObject(m_texture_object));
				if (mp_texture_array) CudaErrorCheck(cudaFreeArray(mp_texture_array));

				m_texture_object = 0ull;
				mp_texture_array = nullptr;
			}


		public:
			template <typename hTextureBuffer_t>
			__host__ void Reconstruct(
				const CudaWorld& hCudaWorld,
				const Handle<hTextureBuffer_t>& hTextureBuffer,
				cudaStream_t& update_stream)
			{
				if (!hTextureBuffer->GetStateRegister().IsModified()) return;

				if (mp_texture_array == nullptr)
				{	// no texture memory allocated

					// texture memory allocation
					cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<CudaVectorType<T>::type>();
					CudaErrorCheck(cudaMallocArray(
						&mp_texture_array,
						&channel_desc,
						hTextureBuffer->GetBitmap().GetWidth(), hTextureBuffer->GetBitmap().GetHeight()));

					// copy host texture buffer data to device array
					CudaErrorCheck(cudaMemcpyToArray(
						mp_texture_array,
						0u, 0u, hTextureBuffer->GetBitmap().GetMapAddress(),
						hTextureBuffer->GetBitmap().GetWidth() * hTextureBuffer->GetBitmap().GetHeight() * sizeof(CudaVectorType<T>::type),
						cudaMemcpyKind::cudaMemcpyHostToDevice));

					// specify resource description
					std::memset(&m_res_desc, 0, sizeof(cudaResourceDesc));
					m_res_desc.resType = cudaResourceType::cudaResourceTypeArray;
					m_res_desc.res.array.array = mp_texture_array;

					// specify texture object parameters
					std::memset(&m_texture_desc, 0, sizeof(cudaTextureDesc));

					// address mode
					switch (hTextureBuffer->GetAddressMode())
					{
						case hTextureBuffer_t::AddressMode::Wrap:
							m_texture_desc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeWrap;
							m_texture_desc.addressMode[1] = cudaTextureAddressMode::cudaAddressModeWrap;
							break;
						case hTextureBuffer_t::AddressMode::Clamp:
							m_texture_desc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeClamp;
							m_texture_desc.addressMode[1] = cudaTextureAddressMode::cudaAddressModeClamp;
							break;
						case hTextureBuffer_t::AddressMode::Mirror:
							m_texture_desc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeMirror;
							m_texture_desc.addressMode[1] = cudaTextureAddressMode::cudaAddressModeMirror;
							break;
						case hTextureBuffer_t::AddressMode::Border:
							m_texture_desc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeBorder;
							m_texture_desc.addressMode[1] = cudaTextureAddressMode::cudaAddressModeBorder;
							break;
					}

					// filter mode
					switch (hTextureBuffer->GetFilterMode())
					{
						case hTextureBuffer_t::FilterMode::Point:
							m_texture_desc.filterMode = cudaTextureFilterMode::cudaFilterModePoint;
							break;
						case hTextureBuffer_t::FilterMode::Linear:
							//m_texture_desc.filterMode = cudaTextureFilterMode::cudaFilterModeLinear;
							m_texture_desc.filterMode = cudaTextureFilterMode::cudaFilterModePoint;
							break;
					}

					if (is_integral_v<T> && normalized_read)
						m_texture_desc.readMode = cudaTextureReadMode::cudaReadModeNormalizedFloat;
					else
						m_texture_desc.readMode = cudaTextureReadMode::cudaReadModeElementType;
					m_texture_desc.normalizedCoords = 1;

					// create texture object
					CudaErrorCheck(cudaCreateTextureObject(
						&m_texture_object,
						&m_res_desc,
						&m_texture_desc, nullptr));
				}
				else
				{
					// get texture array info (width and height)
					cudaExtent array_info;
					CudaErrorCheck(cudaArrayGetInfo(nullptr, &array_info, nullptr, mp_texture_array));

					if (array_info.width * array_info.height !=
						hTextureBuffer->GetBitmap().GetWidth() * hTextureBuffer->GetBitmap().GetHeight())
					{	// size of hTextureBuffer and cuda texture don't match

						// free array
						CudaErrorCheck(cudaFreeArray(mp_texture_array));

						// allocate array of new size
						cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<CudaVectorType<T>::type>();
						CudaErrorCheck(cudaMallocArray(
							&mp_texture_array,
							&channel_desc,
							hTextureBuffer->GetBitmap().GetWidth(), hTextureBuffer->GetBitmap().GetHeight()));
						m_res_desc.res.array.array = mp_texture_array;

						// copy hTextureBuffer data to device array
						CudaErrorCheck(cudaMemcpyToArray(
							mp_texture_array,
							0u, 0u, hTextureBuffer->GetBitmap().GetMapAddress(),
							hTextureBuffer->GetBitmap().GetWidth() * hTextureBuffer->GetBitmap().GetHeight() * sizeof(CudaVectorType<T>::type),
							cudaMemcpyKind::cudaMemcpyHostToDevice));
					}
					else
					{	// Everything does match so do asynchronous texture update

						// TODO: get host pinned memory for asynchronous copying

						CudaErrorCheck(cudaMemcpyToArrayAsync(
							mp_texture_array,
							0u, 0u, hTextureBuffer->GetBitmap().GetMapAddress(),
							hTextureBuffer->GetBitmap().GetWidth() *
							hTextureBuffer->GetBitmap().GetHeight() *
							sizeof(CudaVectorType<T>::type),
							cudaMemcpyKind::cudaMemcpyHostToDevice, update_stream));
						CudaErrorCheck(cudaStreamSynchronize(update_stream));
					}
				}

				hTextureBuffer->GetStateRegister().MakeUnmodified();
			}


			using return_type = typename ReadType<T, normalized_read>::type;
			using cuda_type = typename CudaVectorType<return_type>::type;
			__device__ return_type Fetch(const CudaTexcrd& texcrd) const
			{
				cuda_type value;
				#if defined(__CUDACC__)	
				value = tex2D<cuda_type>(m_texture_object, texcrd.x, texcrd.y);
				#endif
				return CudaVectorTypeConvert<cuda_type, return_type>(value);
			}
		};

		typedef CudaTextureBuffer<ColorU, true> CudaTexture;
		typedef CudaTextureBuffer<float, false> CudaEmittanceMap;
	}
}

#endif