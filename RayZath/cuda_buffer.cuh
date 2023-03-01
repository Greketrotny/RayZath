#ifndef CUDA_BUFFER_CUH
#define CUDA_BUFFER_CUH

#include "cuda_exception.hpp"
#include "cuda_render_parts.cuh"

namespace RayZath::Cuda
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
	template<> struct CudaVectorType<uint8_t>
	{
		using type = uchar1;
	};

	template<typename T1, typename T2>
	__device__ __inline__ T2 cudaVectorTypeConvert(const T1& t1)
	{
		return T2(t1);
	}
	template<> __device__ __inline__ float4 cudaVectorTypeConvert(const ColorF& c)
	{
		return float4{c.red, c.green, c.blue, c.alpha};
	}
	template<> __device__ __inline__ ColorF cudaVectorTypeConvert(const float4& c)
	{
		return ColorF(c.x, c.y, c.z, c.w);
	}
	template<> __device__ __inline__ uchar4 cudaVectorTypeConvert(const ColorU& c)
	{
		return uchar4{c.red, c.green, c.blue, c.alpha};
	}
	template<> __device__ __inline__ ColorU cudaVectorTypeConvert(const uchar4& c)
	{
		return ColorU(c.x, c.y, c.z, c.w);
	}
	template<> __device__ __inline__ float4 cudaVectorTypeConvert(const vec3f& v)
	{
		return make_float4(v.x, v.y, v.z, 0.0f);
	}
	template<> __device__ __inline__ vec3f cudaVectorTypeConvert(const float4& v)
	{
		return vec3f(v.x, v.y, v.z);
	}
	template<> __device__ __inline__ uint16_t cudaVectorTypeConvert(const ushort1& v)
	{
		return v.x;
	}
	template<> __device__ __inline__ ushort1 cudaVectorTypeConvert(const uint16_t& v)
	{
		return make_ushort1(v);
	}
	template<> __device__ __inline__ uint8_t cudaVectorTypeConvert(const uchar1& v)
	{
		return v.x;
	}
	template <> __device__ __inline__ uchar1 cudaVectorTypeConvert(const uint8_t& v)
	{
		return make_uchar1(v);
	}


	template <typename T, typename... Types>
	constexpr bool is_any_of_v = std::disjunction_v<std::is_same<T, Types>...>;
	template <typename T>
	constexpr bool is_integral_v = is_any_of_v<T,
		ColorU, vec2ui16, vec2ui32, uint8_t, uint16_t>;

	template <typename T, bool F>
	struct ReadType;
	template <> struct ReadType<ColorU, true> { using type = ColorF; };
	template <> struct ReadType<ColorU, false> { using type = ColorU; };
	template <> struct ReadType<float, true> { using type = float; };
	template <> struct ReadType<float, false> { using type = float; };
	template <> struct ReadType<uint8_t, true> { using type = float; };
	template <> struct ReadType<uint8_t, false> { using type = uint8_t; };



	template <typename T>
	struct SurfaceBuffer
	{
	private:
		vec2ui32 m_resolution;
		cudaSurfaceObject_t m_so;
		cudaArray* mp_array;

	public:
		__host__ __device__ SurfaceBuffer(const SurfaceBuffer&) = delete;
		__host__ __device__ SurfaceBuffer(SurfaceBuffer&&) = delete;
		__host__ SurfaceBuffer(
			const vec2ui32 resolution = vec2ui32(1u, 1u))
			: m_resolution(resolution)
			, m_so(0u)
			, mp_array(nullptr)
		{
			if (m_resolution.x == 0u) m_resolution.x = 1u;
			if (m_resolution.y == 0u) m_resolution.y = 1u;
			allocate();
		}
		__host__ ~SurfaceBuffer()
		{
			deallocate();
		}

	public:
		__host__ __device__ SurfaceBuffer& operator=(const SurfaceBuffer&) = delete;
		__host__ __device__ SurfaceBuffer& operator=(SurfaceBuffer&&) = delete;

	public:
		__host__ void reset(const vec2ui32 resolution)
		{
			deallocate();
			m_resolution.x = std::max(resolution.x, 1u);
			m_resolution.y = std::max(resolution.y, 1u);
			allocate();
		}
		__host__ cudaArray* getCudaArray()
		{
			return mp_array;
		}
	private:
		__host__ void allocate()
		{
			if (m_so != 0u || mp_array != nullptr) return;

			// allocate cuda array
			auto cd = cudaCreateChannelDesc<CudaVectorType<T>::type>();
			RZAssertCoreCUDA(cudaMallocArray(
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
			RZAssertCoreCUDA(cudaCreateSurfaceObject(&m_so, &rd));
		}
		__host__ void deallocate()
		{
			if (m_so)
			{
				RZAssertCoreCUDA(cudaDestroySurfaceObject(m_so));
				m_so = 0u;
			}
			if (mp_array)
			{
				RZAssertCoreCUDA(cudaFreeArray(mp_array));
				mp_array = nullptr;
			}
		}

	public:
		__device__ __inline__ void SetValue(
			[[maybe_unused]] const vec2ui32 point,
			[[maybe_unused]] const T& value)
		{
			#if defined(__CUDACC__)
			surf2Dwrite<CudaVectorType<T>::type>(
				cudaVectorTypeConvert<T, CudaVectorType<T>::type>(value),
				m_so, point.x * sizeof(CudaVectorType<T>::type), point.y);
			#endif
		}
		__device__ __inline__ T GetValue([[maybe_unused]] const vec2ui32 point)
		{
			typename CudaVectorType<T>::type value;
			#if defined(__CUDACC__)
			surf2Dread<CudaVectorType<T>::type>(
				&value, m_so, point.x * sizeof(CudaVectorType<T>::type), point.y);
			#endif
			return cudaVectorTypeConvert<decltype(value), T>(value);
		}
		__device__ __inline__ void AppendValue(
			const vec2ui32 point,
			const T& value)
		{
			T v = GetValue(point);
			v += value;
			SetValue(point, v);
		}
	};


	template <typename T>
	struct GlobalBuffer
	{
	private:
		vec2ui32 m_resolution;
		size_t m_pitch;
		T* mp_array;


	public:
		__host__ __device__ GlobalBuffer(const GlobalBuffer&) = delete;
		__host__ __device__ GlobalBuffer(GlobalBuffer&&) = delete;
		__host__ GlobalBuffer(
			const vec2ui32& resolution = vec2ui32(0u, 0u))
			: m_resolution(resolution)
			, m_pitch(resolution.x)
			, mp_array(nullptr)
		{
			allocate();
		}
		__host__ ~GlobalBuffer()
		{
			deallocate();
		}

	public:
		__host__ __device__ GlobalBuffer& operator=(const GlobalBuffer&) = delete;
		__host__ __device__ GlobalBuffer& operator=(GlobalBuffer&&) = delete;

	public:
		__host__ void reset(const vec2ui32& resolution)
		{
			deallocate();
			m_resolution.x = std::max(resolution.x, 1u);
			m_resolution.y = std::max(resolution.y, 1u);
			allocate();
		}
		__host__ T* GetDataPtr()
		{
			return mp_array;
		}
	private:
		__host__ void allocate()
		{
			if (mp_array != nullptr) return;
			RZAssertCoreCUDA(cudaMallocPitch(
				&mp_array,
				&m_pitch,
				m_resolution.x * sizeof(T), m_resolution.y));
		}
		__host__ void deallocate()
		{
			if (mp_array)
			{
				RZAssertCoreCUDA(cudaFree(mp_array));
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
			const vec2ui32& point,
			const T& value)
		{
			Value(point) = value;
		}
		__device__ __inline__ const T& GetValue(
			const vec2ui32& point) const
		{
			return Value(point);
		}
		__device__ __inline__ void AppendValue(
			const vec2ui32& point,
			const T& value)
		{
			Value(point) += value;
		}
	};


	template <typename T, bool normalized_read = true>
	struct TextureBuffer
	{
	private:
		cudaArray* mp_texture_array = nullptr;
		cudaTextureObject_t m_texture_object{};
		vec2f m_scale{1.0f};
		float m_rotation = 1.0f;
		vec2f m_translation{0.0f};


	public:
		__host__ ~TextureBuffer()
		{
			destroy();
		}


		template <typename hTextureBuffer_t>
		__host__ void reconstruct(
			[[maybe_unused]] const World& hCudaWorld,
			const RayZath::Engine::Handle<hTextureBuffer_t>& hTextureBuffer,
			cudaStream_t& update_stream)
		{
			if (!hTextureBuffer->stateRegister().IsModified()) return;

			m_scale = hTextureBuffer->scale();
			m_rotation = hTextureBuffer->rotation().value();
			m_translation = hTextureBuffer->translation();

			const bool has_to_recreate = [&]() {
				if (mp_texture_array != nullptr)
				{
					cudaExtent array_extent;
					RZAssertCoreCUDA(cudaArrayGetInfo(nullptr, &array_extent, nullptr, mp_texture_array));

					const auto same_resolution =
						array_extent.width == hTextureBuffer->bitmap().GetWidth() &&
						array_extent.height == hTextureBuffer->bitmap().GetHeight();
					return !same_resolution;
				}
				return true;
			}();

			if (has_to_recreate)
			{
				destroy();

				// texture memory allocation
				const cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<CudaVectorType<T>::type>();
				RZAssertCoreCUDA(cudaMallocArray(
					&mp_texture_array,
					&channel_desc,
					hTextureBuffer->bitmap().GetWidth(), hTextureBuffer->bitmap().GetHeight()));

				// copy host texture buffer data to device array
				RZAssertCoreCUDA(cudaMemcpyToArray(
					mp_texture_array,
					0u, 0u, hTextureBuffer->bitmap().GetMapAddress(),
					hTextureBuffer->bitmap().GetWidth() * hTextureBuffer->bitmap().GetHeight() * sizeof(CudaVectorType<T>::type),
					cudaMemcpyKind::cudaMemcpyHostToDevice));

				// specify resource description
				cudaResourceDesc resource_desc{};
				resource_desc.resType = cudaResourceType::cudaResourceTypeArray;
				resource_desc.res.array.array = mp_texture_array;

				cudaTextureDesc texture_desc{};
				// address mode
				switch (hTextureBuffer->addressMode())
				{
					case hTextureBuffer_t::AddressMode::Wrap:
						texture_desc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeWrap;
						texture_desc.addressMode[1] = cudaTextureAddressMode::cudaAddressModeWrap;
						break;
					case hTextureBuffer_t::AddressMode::Clamp:
						texture_desc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeClamp;
						texture_desc.addressMode[1] = cudaTextureAddressMode::cudaAddressModeClamp;
						break;
					case hTextureBuffer_t::AddressMode::Mirror:
						texture_desc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeMirror;
						texture_desc.addressMode[1] = cudaTextureAddressMode::cudaAddressModeMirror;
						break;
					case hTextureBuffer_t::AddressMode::Border:
						texture_desc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeBorder;
						texture_desc.addressMode[1] = cudaTextureAddressMode::cudaAddressModeBorder;
						break;
				}

				// filter mode
				if ((!is_integral_v<T> || normalized_read) &&
					hTextureBuffer->filterMode() == hTextureBuffer_t::FilterMode::Linear)
				{
					texture_desc.filterMode = cudaTextureFilterMode::cudaFilterModeLinear;
				}
				else
				{
					texture_desc.filterMode = cudaTextureFilterMode::cudaFilterModePoint;
				}

				if constexpr (is_integral_v<T> && normalized_read)
					texture_desc.readMode = cudaTextureReadMode::cudaReadModeNormalizedFloat;
				else
					texture_desc.readMode = cudaTextureReadMode::cudaReadModeElementType;
				texture_desc.normalizedCoords = 1;

				// create texture object
				RZAssertCoreCUDA(cudaCreateTextureObject(
					&m_texture_object,
					&resource_desc,
					&texture_desc, nullptr));
			}
			else
			{
				// TODO: perform asynchronous copying through host pinned memory
				RZAssertCoreCUDA(cudaMemcpyToArrayAsync(
					mp_texture_array,
					0u, 0u, 
					hTextureBuffer->bitmap().GetMapAddress(),
					hTextureBuffer->bitmap().GetWidth() * hTextureBuffer->bitmap().GetHeight() * sizeof(CudaVectorType<T>::type),
					cudaMemcpyKind::cudaMemcpyHostToDevice, update_stream));
				RZAssertCoreCUDA(cudaStreamSynchronize(update_stream));
			}

			hTextureBuffer->stateRegister().MakeUnmodified();
		}


		using return_type = typename ReadType<T, normalized_read>::type;
		using cuda_type = typename CudaVectorType<return_type>::type;
		__device__ return_type fetch(Texcrd texcrd) const
		{
			texcrd += m_translation;
			texcrd.Rotate(m_rotation);
			texcrd *= m_scale;

			cuda_type value;
			#if defined(__CUDACC__)	
			value = tex2D<cuda_type>(m_texture_object, texcrd.x, 1.0f - texcrd.y);
			#endif
			return cudaVectorTypeConvert<cuda_type, return_type>(value);
		}
	private:
		void destroy()
		{
			if (m_texture_object) RZAssertCoreCUDA(cudaDestroyTextureObject(m_texture_object));
			if (mp_texture_array) RZAssertCoreCUDA(cudaFreeArray(mp_texture_array));

			m_texture_object = 0;
			mp_texture_array = nullptr;
		}
	};

	typedef TextureBuffer<ColorU, true> Texture;
	typedef TextureBuffer<ColorU, true> NormalMap;
	typedef TextureBuffer<uint8_t, true> MetalnessMap;
	typedef TextureBuffer<uint8_t, true> RoughnessMap;
	typedef TextureBuffer<float, false> EmissionMap;
}

#endif