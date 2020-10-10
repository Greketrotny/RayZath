#ifndef CUDA_RENDER_PARTS_CUH
#define CUDA_RENDER_PARTS_CUH

#include "cuda_engine_parts.cuh"
#include "render_object.h"
//#include "rzexception.h"
#include "vec3.h"
#include "color.h"

#include "render_parts.h"

namespace RayZath
{
	template <typename T> struct cudaVec3
	{
	public:
		T x, y, z;

	public:
		__host__ __device__ cudaVec3()
			: x(0.0f)
			, y(0.0f)
			, z(0.0f)
		{}
		__host__ __device__ cudaVec3(const cudaVec3& V)
			: x(V.x)
			, y(V.y)
			, z(V.z)
		{}
		__host__ __device__ cudaVec3(cudaVec3&& V)
			: x(V.x)
			, y(V.y)
			, z(V.z)
		{}
		__host__ __device__ cudaVec3(T x, T y, T z)
			: x(x)
			, y(y)
			, z(z)
		{}
		__host__ cudaVec3(const Math::vec3<T>& v)
		{
			this->x = v.x;
			this->y = v.y;
			this->z = v.z;
		}
		__host__ __device__ ~cudaVec3()
		{}


	public:
		__device__ __inline__  static T DotProduct(const cudaVec3& V1, const cudaVec3& V2)
		{
			return V1.x * V2.x + V1.y * V2.y + V1.z * V2.z;
		}
		__device__ __inline__  static cudaVec3 CrossProduct(const cudaVec3& V1, const cudaVec3& V2)
		{
			return cudaVec3(
				V1.y * V2.z - V1.z * V2.y,
				V1.z * V2.x - V1.x * V2.z,
				V1.x * V2.y - V1.y * V2.x);
		}
		__device__ __inline__  static T Similarity(const cudaVec3& V1, const cudaVec3& V2)
		{
			return DotProduct(V1, V2) / (V1.Magnitude() * V2.Magnitude());
		}
		__device__ __inline__  static T Distance(const cudaVec3& V1, const cudaVec3& V2)
		{
			return (V1 - V2).Magnitude();
		}
		__device__ __inline__  static cudaVec3 Normalize(const cudaVec3& V)
		{
			cudaVec3<T> normalized = V;
			normalized.Normalize();
			return normalized;
		}
		__device__ __inline__  static cudaVec3 Reverse(const cudaVec3& V)
		{
			return cudaVec3(
				-V.x,
				-V.y,
				-V.z);
		}


	public:
		__device__ __inline__ T DotProduct(const cudaVec3& V)
		{
			return (this->x * V.x + this->y * V.y + this->z * V.z);
		}
		__device__ __inline__ void CrossProduct(const cudaVec3& V)
		{
			this->x = this->y * V.z - this->z * V.y;
			this->y = this->z * V.x - this->x * V.z;
			this->z = this->x * V.y - this->y * V.x;
		}
		__device__ __inline__ T Similarity(const cudaVec3& V)
		{
			return this->DotProduct(V) / (this->Magnitude() * V.Magnitude());
		}
		__device__ __inline__ T Distance(const cudaVec3& V)
		{
			return (*this - V).Magnitude();
		}
		__device__ __inline__ void Normalize()
		{
			T scalar = 1.0f / Magnitude();
			x *= scalar;
			y *= scalar;
			z *= scalar;
		}
		__device__ __inline__ void Reverse()
		{
			x = -x;
			y = -y;
			z = -z;
		}
		__device__ __inline__ void RotateX(T angle)
		{
			#if defined(__CUDACC__)
			T sina = __sinf(angle);
			T cosa = __cosf(angle);
			T newY = y * cosa + z * sina;
			z = y * -sina + z * cosa;
			y = newY;
			#endif
		}
		__device__ __inline__ void RotateY(T angle)
		{
			#if defined(__CUDACC__)
			T sina = __sinf(angle);
			T cosa = __cosf(angle);
			T newX = x * cosa - z * sina;
			z = x * sina + z * cosa;
			x = newX;
			#endif
		}
		__device__ __inline__ void RotateZ(T angle)
		{
			#if defined(__CUDACC__)
			T sina = __sinf(angle);
			T cosa = __cosf(angle);
			T newX = x * cosa + y * sina;
			y = x * -sina + y * cosa;
			x = newX;
			#endif
		}

		__device__ __inline__ void RotateXYZ(const cudaVec3& rot)
		{
			#if defined(__CUDACC__)
			// x rotation
			T sina = __sinf(rot.x);	// sin(angle)
			T cosa = __cosf(rot.x);	// cos(angle)
			T newValue = y * cosa + z * sina;	// new y
			z = y * -sina + z * cosa;			// new z
			y = newValue;

			// y rotation
			sina = __sinf(rot.y);
			cosa = __cosf(rot.y);
			newValue = x * cosa - z * sina;	// new x
			z = x * sina + z * cosa;		// new z
			x = newValue;

			// z rotation
			sina = __sinf(rot.z);
			cosa = __cosf(rot.z);
			newValue = x * cosa + y * sina;	// new x
			y = x * -sina + y * cosa;		// new y
			x = newValue;
			#endif
		}
		__device__ __inline__ void RotateZYX(const cudaVec3& rot)
		{
			#if defined(__CUDACC__)
			// z rotation
			T sina = __sinf(rot.z);
			T cosa = __cosf(rot.z);
			T newValue = x * cosa + y * sina;
			y = x * -sina + y * cosa;
			x = newValue;

			// y rotation
			sina = __sinf(rot.y);
			cosa = __cosf(rot.y);
			newValue = x * cosa - z * sina;
			z = x * sina + z * cosa;
			x = newValue;

			// x rotation
			sina = __sinf(rot.x);
			cosa = __cosf(rot.x);
			newValue = y * cosa + z * sina;
			z = y * -sina + z * cosa;
			y = newValue;
			#endif
		}


	public:
		__host__ __device__ cudaVec3 operator-() const
		{
			return cudaVec3(-this->x, -this->y, -this->z);
		}
		__host__ __device__ cudaVec3 operator+(const cudaVec3& V) const
		{
			return cudaVec3(this->x + V.x, this->y + V.y, this->z + V.z);
		}
		__host__ __device__ cudaVec3 operator-(const cudaVec3& V) const
		{
			return cudaVec3(this->x - V.x, this->y - V.y, this->z - V.z);
		}
		__host__ __device__ cudaVec3 operator*(T scalar) const
		{
			return cudaVec3(this->x * scalar, this->y * scalar, this->z * scalar);
		}
		__host__ __device__ cudaVec3 operator*(const cudaVec3& scalar) const
		{
			return cudaVec3(this->x * scalar.x, this->y * scalar.y, this->z * scalar.z);
		}
		__host__ __device__ cudaVec3 operator/(T scalar) const
		{
			return cudaVec3(this->x / scalar, this->y / scalar, this->z / scalar);
		}
		__host__ __device__ cudaVec3 operator/(const cudaVec3& scalar) const
		{
			return cudaVec3(this->x / scalar.x, this->y / scalar.y, this->z / scalar.z);
		}
		__host__ __device__ cudaVec3& operator+=(const cudaVec3& V)
		{
			this->x += V.x;
			this->y += V.y;
			this->z += V.z;
			return *this;
		}
		__host__ __device__ cudaVec3& operator-=(const cudaVec3& V)
		{
			this->x -= V.x;
			this->y -= V.y;
			this->z -= V.z;
			return *this;
		}
		__host__ __device__ cudaVec3& operator*=(T scalar)
		{
			this->x *= scalar;
			this->y *= scalar;
			this->z *= scalar;
			return *this;
		}
		__host__ __device__ cudaVec3& operator*=(const cudaVec3& scalar)
		{
			this->x *= scalar.x;
			this->y *= scalar.y;
			this->z *= scalar.z;
			return *this;
		}
		__host__ __device__ cudaVec3& operator/=(T scalar)
		{
			this->x /= scalar;
			this->y /= scalar;
			this->z /= scalar;
			return *this;
		}
		__host__ __device__ cudaVec3& operator/=(const cudaVec3& scalar)
		{
			this->x /= scalar.x;
			this->y /= scalar.y;
			this->z /= scalar.z;
			return *this;
		}
		__host__ __device__ cudaVec3& operator=(const cudaVec3& V)
		{
			x = V.x;
			y = V.y;
			z = V.z;
			return *this;
		}
		__host__ __device__ cudaVec3& operator=(cudaVec3&& V) noexcept
		{
			x = V.x;
			y = V.y;
			z = V.z;
			return *this;
		}
		__host__ cudaVec3& operator=(const Math::vec3<T> v)
		{
			x = v.x;
			y = v.y;
			z = v.z;
			return *this;
		}


	public:
		__device__ void SetValues(T x, T y, T z)
		{
			this->x = x;
			this->y = y;
			this->z = z;
		}
		__device__ T Magnitude() const
		{
			return (T)sqrtf(x * x + y * y + z * z);
		}
	};

	template <typename T = unsigned char> class CudaColor
	{
	};
	template<> class CudaColor<unsigned char>
	{
	public:
		unsigned char blue, green, red, alpha;
		// this order ^^^^^^^^^^^^^^^^^^^^^^^ is very important!


	public:
		__device__ CudaColor()
		{
			red = 255;
			green = 255;
			blue = 255;
			alpha = 255;
		}
		__host__ CudaColor(const CudaColor<unsigned char>& color)
			: red(color.red)
			, green(color.green)
			, blue(color.blue)
			, alpha(color.alpha)
		{}
		__device__ CudaColor(
			const unsigned char& red,
			const unsigned char& green,
			const unsigned char& blue,
			const unsigned char& alpha = 0xFF)
			: red(red)
			, green(green)
			, blue(blue)
			, alpha(alpha)
		{}
		__host__ CudaColor(const Graphics::Color& color)
			: red(color.GetR())
			, green(color.GetG())
			, blue(color.GetB())
			, alpha(color.GetA())
		{}
		__host__ __device__ ~CudaColor()
		{}


	public:
		__host__ __device__ CudaColor<unsigned char>& operator=(const CudaColor<unsigned char>& color)
		{
			this->red = color.red;
			this->green = color.green;
			this->blue = color.blue;
			this->alpha = color.alpha;
			return *this;
		}
		__host__ CudaColor<unsigned char>& operator=(const Graphics::Color& color)
		{
			this->red = color.GetR();
			this->green = color.GetG();
			this->blue = color.GetB();
			this->alpha = color.GetA();
			return *this;
		}
		__device__ CudaColor<unsigned char>& operator*=(const float& factor)
		{
			this->red = static_cast<unsigned char>(this->red * factor);
			this->green = static_cast<unsigned char>(this->green * factor);
			this->blue = static_cast<unsigned char>(this->blue * factor);
			this->alpha = static_cast<unsigned char>(this->alpha * factor);
			return *this;
		}
		__device__ CudaColor<unsigned char> operator*(const float& factor) const
		{
			return CudaColor<unsigned char>(
				static_cast<unsigned char>(this->red * factor),
				static_cast<unsigned char>(this->green * factor),
				static_cast<unsigned char>(this->blue * factor),
				static_cast<unsigned char>(this->alpha * factor));
		}


	public:
		__device__ static CudaColor<unsigned char> BlendAverage(const CudaColor<unsigned char>& color1, const CudaColor<unsigned char>& color2)
		{
			return CudaColor<unsigned char>(
				(color1.red + color2.red) / 2,
				(color1.green + color2.green) / 2,
				(color1.blue + color2.blue) / 2,
				(color1.alpha + color2.alpha) / 2);
		}
		__device__ static CudaColor<unsigned char> BlendAverage(const CudaColor<unsigned char>& color1, const CudaColor<unsigned char>& color2, const unsigned char balance)
		{
			return CudaColor<unsigned char>(
				(color1.red * balance + color2.red * (255u - balance)) / 255u,
				(color1.green * balance + color2.green * (255u - balance)) / 255u,
				(color1.blue * balance + color2.blue * (255u - balance)) / 255u,
				(color1.alpha * balance + color2.blue * (255u - balance)) / 255u);
		}
		__device__ static CudaColor<unsigned char> BlendProduct(const CudaColor<unsigned char>& color1, const CudaColor<unsigned char>& color2)
		{
			return CudaColor<unsigned char>(
				(color1.red * color2.red) / 255u,
				(color1.green * color2.green) / 255u,
				(color1.blue * color2.blue) / 255u,
				(color1.alpha * color2.alpha) / 255u);
		}


	public:
		__device__ void BlendAverage(const CudaColor<unsigned char>& color)
		{
			this->red = (this->red + color.red) / 2;
			this->green = (this->green + color.green) / 2;
			this->blue = (this->blue + color.blue) / 2;
			this->alpha = (this->alpha + color.alpha) / 2;
		}
		__device__ void BlendAverage(const CudaColor<unsigned char>& color, const unsigned char balance)
		{
			this->red = (this->red * (255u - balance) + color.red * balance) / 255u;
			this->green = (this->green * (255u - balance) + color.green * balance) / 255u;
			this->blue = (this->blue * (255u - balance) + color.blue * balance) / 255u;
			this->alpha = (this->alpha * (255u - balance) + color.alpha * balance) / 255u;
		}
		__device__ void BlendProduct(const CudaColor<unsigned char>& color)
		{
			*this = CudaColor<unsigned char>::BlendProduct(*this, color);
		}


	public:
		__device__ void SetColor(
			const unsigned char& r,
			const unsigned char& g,
			const unsigned char& b,
			const unsigned char& a = 0xFF)
		{
			red = r;
			green = g;
			blue = b;
			alpha = a;
		}
	};
	template<> class CudaColor<float>
	{
	public:
		float red, green, blue, alpha;


	public:
		__host__ __device__ CudaColor()
		{
			red = 1.0f;
			green = 1.0f;
			blue = 1.0f;
			alpha = 1.0f;
		}
		__host__ __device__ CudaColor(const CudaColor<float>& color)
			: red(color.red)
			, green(color.green)
			, blue(color.blue)
			, alpha(color.alpha)
		{}
		__host__ __device__ CudaColor(const float& red, const float& green, const float& blue, const float& alpha = 1.0f)
			: red(red)
			, green(green)
			, blue(blue)
			, alpha(alpha)
		{
		}
		__host__ CudaColor(const Graphics::Color& color)
			: red(color.GetR() / 255.0f)
			, green(color.GetG() / 255.0f)
			, blue(color.GetB() / 255.0f)
			, alpha(color.GetA() / 255.0f)
		{}
		__host__ __device__ ~CudaColor() {}


	public:
		__device__ CudaColor<float> operator*(const float& factor) const
		{
			return CudaColor<float>(
				this->red * factor,
				this->green * factor,
				this->blue * factor,
				this->alpha * factor);
		}
		__device__ CudaColor<float> operator+(const CudaColor<float>& color) const
		{
			return CudaColor<float>(
				this->red + color.red,
				this->green + color.green,
				this->blue + color.blue,
				this->alpha + color.alpha);
		}
		__device__ CudaColor<float> operator/(float factor) const
		{
			factor = 1.0f / factor;
			return CudaColor<float>(
				this->red * factor,
				this->green * factor,
				this->blue * factor,
				this->alpha * factor);
		}
		__host__ __device__ CudaColor<float>& operator=(const CudaColor<float>& color)
		{
			this->red = color.red;
			this->green = color.green;
			this->blue = color.blue;
			this->alpha = color.alpha;
			return *this;
		}
		__host__ CudaColor<float>& operator=(const Graphics::Color& color)
		{
			this->red = color.GetR() / 255.0f;
			this->green = color.GetG() / 255.0f;
			this->blue = color.GetB() / 255.0f;
			this->alpha = color.GetA() / 255.0f;
			return *this;
		}
		__device__ CudaColor<float>& operator*=(const float& factor)
		{
			this->red *= factor;
			this->green *= factor;
			this->blue *= factor;
			this->alpha *= factor;
			return *this;
		}
		__device__ CudaColor<float>& operator+=(const CudaColor<float>& color)
		{
			this->red += color.red;
			this->green += color.green;
			this->blue += color.blue;
			this->alpha += color.alpha;
			return *this;
		}
		__device__ CudaColor<float>& operator/=(float factor)
		{
			factor = 1.0f / factor;
			this->red *= factor;
			this->green *= factor;
			this->blue *= factor;
			this->alpha *= factor;
			return *this;
		}


	public:
		__device__ static CudaColor<float> BlendAverage(
			const CudaColor<float>& color1,
			const CudaColor<float>& color2,
			const float& balance)
		{
			return CudaColor<float>(
				color1.red * balance + color2.red * (1.0f - balance),
				color1.green * balance + color2.green * (1.0f - balance),
				color1.blue * balance + color2.blue * (1.0f - balance),
				color1.alpha * balance + color2.alpha * (1.0f - balance));
		}
		__device__ static CudaColor<float> BlendProduct(
			const CudaColor<float>& color1,
			const CudaColor<float>& color2)
		{
			return CudaColor<float>(
				color1.red * color2.red,
				color1.green * color2.green,
				color1.blue * color2.blue,
				color1.alpha * color2.alpha);
		}


	public:
		__device__ void BlendAverage(const CudaColor<float>& color)
		{
			this->red = (this->red + color.red) / 2.0f;
			this->green = (this->green + color.green) / 2.0f;
			this->blue = (this->blue + color.blue) / 2.0f;
			this->alpha = (this->alpha + color.alpha) / 2.0f;
		}
		__device__ void BlendAverage(const CudaColor<float>& color, const float& balance)
		{
			this->red = (this->red * balance + color.red * (1.0f - balance));
			this->green = (this->green * balance + color.green * (1.0f - balance));
			this->blue = (this->blue * balance + color.blue * (1.0f - balance));
			this->alpha = (this->alpha * balance + color.alpha * (1.0f - balance));
		}
		__device__ void BlendProduct(const CudaColor<float>& color)
		{
			this->red *= color.red;
			this->green *= color.green;
			this->blue *= color.blue;
			this->alpha *= color.alpha;
		}


	public:
		__host__ __device__ void SetColor(
			const float& r,
			const float& g,
			const float& b,
			const float& a)
		{
			red = r;
			green = g;
			blue = b;
			alpha = a;
		}
	};

	struct CudaRay;

	// ~~~~~~~~ Helper Functions Declarations ~~~~~~~~
	__device__ __inline__ cudaVec3<float> ReflectVector(
		const cudaVec3<float>& indicent,
		const cudaVec3<float>& normal);
	__device__ __inline__ float RayToPointDistance(
		const CudaRay& ray,
		const cudaVec3<float>& point);

	__device__ __inline__ cudaVec3<float> SampleHemisphere(
		const float r1,
		const float r2,
		const cudaVec3<float>& normal);
	__device__ __inline__ void RandomVectorOnAngularSphere(
		const float& unsigned_random_theta,
		const float& usnigned_random_phi,
		const cudaVec3<float>& normal,
		cudaVec3<float>& randomVector);
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	struct CudaMaterial
	{
		float reflectance;
		float glossiness;

		float transmitance;
		float ior;

		float emitance;

		__host__ __device__ CudaMaterial()
			: reflectance(0.0f)
			, glossiness(0.0f)
			, transmitance(1.0f)
			, ior(1.0f)
			, emitance(0.0f)
		{}
		__host__ __device__ ~CudaMaterial()
		{}

		__host__ CudaMaterial& operator=(const Material& host_material);
		__device__ CudaMaterial& operator=(const CudaMaterial& cuda_material)
		{
			this->reflectance = cuda_material.reflectance;
			this->glossiness = cuda_material.glossiness;
			this->transmitance = cuda_material.transmitance;
			this->ior = cuda_material.ior;
			this->emitance = cuda_material.emitance;
			return *this;
		}
	};

	struct RandomNumbers
	{
	public:
		static constexpr unsigned int s_count = 0x400;
	private:
		float m_unsigned_uniform[s_count];
		float m_signed_uniform[s_count];
		unsigned int m_seed = 0u;

		static HostPinnedMemory s_hpm;


	public:
		__host__ RandomNumbers();
		__host__ RandomNumbers(const RandomNumbers&) = delete;
		__host__ RandomNumbers(RandomNumbers&&) = delete;
		__host__ ~RandomNumbers();


	public:
		__host__ void Reconstruct(cudaStream_t& mirror_stream);
		__device__ __inline__ float GetUnsignedUniform()
		{
			#if defined(__CUDACC__)
			atomicAdd(&m_seed, 1u);
			#endif

			return m_unsigned_uniform[
				(m_seed + threadIdx.y * blockDim.x + threadIdx.x) % RandomNumbers::s_count];
		}
		__device__ __inline__ float GetSignedUniform()
		{
			#if defined(__CUDACC__)
			atomicAdd(&m_seed, 1u);
			#endif

			return m_signed_uniform[
				(m_seed + threadIdx.y * blockDim.x + threadIdx.x) % RandomNumbers::s_count];
		}
	};
	class CudaKernelData
	{
	public:
		unsigned int renderIndex;
		RandomNumbers randomNumbers;


	public:
		__host__ CudaKernelData();
		__host__ CudaKernelData(const CudaKernelData&) = delete;
		__host__ CudaKernelData(CudaKernelData&&) = delete;
		__host__ ~CudaKernelData();


	public:
		__host__ void Reconstruct(
			unsigned int renderIndex,
			cudaStream_t& mirrorStream);
	};

	struct PathNode
	{
	public:
		cudaVec3<float> point;


	private:
		__host__ __device__ PathNode()
		{}
		__host__ __device__ ~PathNode()
		{}

		friend struct TracingPath;
	};
	struct TracingPath
	{
	public:
		static constexpr unsigned int MaxPathDepth = 16u;
		//PathNode pathNodes[MaxPathDepth];
		int currentNodeIndex;
		CudaColor<float> finalColor;

	public:
		__host__ TracingPath()
			: currentNodeIndex(0)
		{}
		__host__ ~TracingPath()
		{
			currentNodeIndex = 0;
		}


	public:
		__device__ __inline__ void ResetPath()
		{
			currentNodeIndex = 0;
			finalColor = CudaColor<float>(0.0f, 0.0f, 0.0f);
		}
		__device__ __inline__ bool NextNodeAvailable()
		{
			return !(currentNodeIndex >= MaxPathDepth - 1u);
		}
		__device__ __inline__ bool FindNextNodeToTrace()
		{
			if (currentNodeIndex >= MaxPathDepth - 1u)
				return false;

			++currentNodeIndex;
			return true;
		}
		__device__ __inline__ CudaColor<float> CalculateFinalColor()
		{
			return finalColor;
		}
		/*__device__ __inline__ PathNode& GetCurrentNode()
		{
			return pathNodes[currentNodeIndex];
		}*/
	};


	struct CudaRay
	{
	public:
		cudaVec3<float> origin;
		cudaVec3<float> direction;
		float length;


	public:
		__device__ CudaRay()
			: length(3.402823466e+38f)
		{}
		__device__ CudaRay(
			const cudaVec3<float>& origin,
			const cudaVec3<float>& direction,
			const float& length = 3.402823466e+38f)
			: origin(origin)
			, direction(direction)
			, length(length)
		{
			this->direction.Normalize();
		}
		__device__ ~CudaRay()
		{}
	};
	struct CudaSceneRay : public CudaRay
	{
	public:
		CudaMaterial material;


	public:
		__device__ CudaSceneRay()
			: CudaRay()
		{}
		__device__ CudaSceneRay(
			const cudaVec3<float>& origin,
			const cudaVec3<float>& direction,
			const CudaMaterial& material,
			const float& length = 3.402823466e+38f)
			: CudaRay(origin, direction, length)
			, material(material)
		{}
		__device__ ~CudaSceneRay()
		{}
	};

	struct RayIntersection
	{
		CudaSceneRay ray;
		cudaVec3<float> point;
		cudaVec3<float> surface_normal;
		cudaVec3<float> mapped_normal;
		CudaColor<float> surface_color;
		CudaMaterial material;

		float bvh_factor = 1.0f;


		__device__ RayIntersection()
		{}
		__device__ ~RayIntersection()
		{}
	};
	struct CudaTriangle;
	struct TriangleIntersection
	{
		CudaRay ray;
		cudaVec3<float> point;
		const CudaTriangle* triangle;
		float b1, b2;

		float bvh_factor = 1.0f;

		__device__ TriangleIntersection()
			: triangle(nullptr)
		{
		}
	};

	struct CudaTexcrd
	{
		float u, v;

		__device__ CudaTexcrd(float u, float v)
			: u(u)
			, v(v)
		{}
		__host__ CudaTexcrd(const Texcrd& T)
			: u(T.u)
			, v(T.v)
		{}
	};
	struct CudaTexture
	{
	public:
		static cudaChannelFormatDesc chanelDesc;
		cudaResourceDesc resDesc;
		cudaTextureDesc textureDesc;
		cudaArray* textureArray;
		cudaTextureObject_t textureObject;


	public:
		__host__ CudaTexture();
		__host__ ~CudaTexture();


	public:
		__host__ void Reconstruct(
			const Texture& hTexture,
			cudaStream_t& mirror_stream);
	};

	struct CudaTriangle
	{
	public:
		cudaVec3<float>* v1, * v2, * v3;
		CudaTexcrd* t1, * t2, * t3;
		cudaVec3<float>* n1, * n2, * n3;
		cudaVec3<float> normal;
		CudaColor<float> color;


	public:
		__host__ CudaTriangle(const Triangle& hostTriangle);
		__host__ ~CudaTriangle();


	public:
		__device__ __inline__ bool RayIntersect(TriangleIntersection& intersection) const
		{
			const cudaVec3<float> edge1 = *v2 - *v1;
			const cudaVec3<float> edge2 = *v3 - *v1;

			const cudaVec3<float> pvec = cudaVec3<float>::CrossProduct(intersection.ray.direction, edge2);

			const float det = (cudaVec3<float>::DotProduct(edge1, pvec));
			if (det > -0.0001f && det < 0.0001f)
				return false;

			const float inv_det = 1.0f / det;

			const cudaVec3<float> tvec = intersection.ray.origin - *v1;
			const float b1 = cudaVec3<float>::DotProduct(tvec, pvec) * inv_det;
			if (b1 < 0.0f || b1 > 1.0f)
				return false;

			const cudaVec3<float> qvec = cudaVec3<float>::CrossProduct(tvec, edge1);

			const float b2 = cudaVec3<float>::DotProduct(intersection.ray.direction, qvec) * inv_det;
			if (b2 < 0.0f || b1 + b2 > 1.0f)
				return false;

			const float t = cudaVec3<float>::DotProduct(edge2, qvec) * inv_det;
			if (t <= 0.0f)
				return false;

			cudaVec3<float> P = intersection.ray.origin + intersection.ray.direction * t;

			float dist = (P - intersection.ray.origin).Magnitude();
			if (dist > intersection.ray.length)
				return false;

			intersection.point = P;
			intersection.ray.length = dist;
			intersection.triangle = this;
			intersection.b1 = b1;
			intersection.b2 = b2;

			return true;
		}
	};


	struct CudaBoundingBox
	{
		cudaVec3<float> min, max;

		__host__ CudaBoundingBox() {}
		__host__ CudaBoundingBox(const BoundingBox& box)
			: min(box.min)
			, max(box.max)
		{}

		__host__ CudaBoundingBox& operator=(const BoundingBox& box)
		{
			this->min = box.min;
			this->max = box.max;
			return *this;
		}

		__device__ __inline__ bool RayIntersection(const CudaRay& ray) const
		{
			float t1 = (min.x - ray.origin.x) / ray.direction.x;
			float t2 = (max.x - ray.origin.x) / ray.direction.x;
			float t3 = (min.y - ray.origin.y) / ray.direction.y;
			float t4 = (max.y - ray.origin.y) / ray.direction.y;
			float t5 = (min.z - ray.origin.z) / ray.direction.z;
			float t6 = (max.z - ray.origin.z) / ray.direction.z;

			auto min = [](const float& a, const float& b)
			{
				return (a < b) ? a : b;
			};
			auto max = [](const float& a, const float& b)
			{
				return (a > b) ? a : b;
			};
			float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
			float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

			return !(tmax < 0.0f || tmin > tmax || tmin > ray.length);
		}
	};

	// ~~~~~~~~ Helper Functions Definitions ~~~~~~~~
	__device__ __inline__ cudaVec3<float> ReflectVector(
		const cudaVec3<float>& vI,
		const cudaVec3<float>& vN)
	{
		return (vN * -2.0f * cudaVec3<float>::DotProduct(vN, vI) + vI);
	}
	__device__ __inline__ float RayToPointDistance(
		const CudaRay& ray,
		const cudaVec3<float>& P)
	{
		// O - ray origin
		// P - specified point
		// vD - ray direction

		cudaVec3<float> vOP = P - ray.origin;
		float dOP = vOP.Magnitude();
		float vOP_dot_vD = cudaVec3<float>::DotProduct(vOP, ray.direction);
		return sqrtf(dOP * dOP - vOP_dot_vD * vOP_dot_vD);
	}

	__device__ __inline__ void LocalCoordinate(
		const cudaVec3<float>& vN,
		cudaVec3<float>& vX,
		cudaVec3<float>& vY)
	{
		bool b = (fabs(vN.x) > fabs(vN.y));
		vX.x = static_cast<float>(!b);
		vX.y = static_cast<float>(b);
		vX.z = 0.0f;

		vY = cudaVec3<float>::CrossProduct(vN, vX);
		vX = cudaVec3<float>::CrossProduct(vN, vY);
	}

	__device__ __inline__ cudaVec3<float> CosineSampleHemisphere(
		const float r1,
		const float r2,
		const cudaVec3<float>& vN)
	{
		// create local coordinate space vectors
		cudaVec3<float> vX, vY;
		LocalCoordinate(vN, vX, vY);

		const float phi = r1 * 6.283185f;
		const float theta = r2;

		// calculate sample direction
		#if defined(__CUDACC__)
		const float sqrt_theta = sqrtf(theta);
		return vX * sqrt_theta * __cosf(phi) + vY * sqrt_theta * __sinf(phi) + vN * sqrtf(1.0f - theta);
		//				  along local x axis		+ along local z axis		+ along normal
		#endif
	}
	__device__ __inline__ cudaVec3<float> SampleSphere(
		const float r1,
		const float r2,
		const cudaVec3<float>& vN)
	{
		// create local coordinate space vectors
		cudaVec3<float> vX, vY;
		LocalCoordinate(vN, vX, vY);

		// calculate phi and theta angles
		const float phi = r1 * 6.283185f;
		const float theta = acosf(1.0f - 2.0f * r2);

		// calculate sample direction
		#if defined(__CUDACC__)
		const float sin_theta = __sinf(theta);
		return vX * sin_theta * __cosf(phi) + vY * sin_theta * __sinf(phi) + vN * __cosf(theta);
		//		along local x axis			+ along local y axis			+ along normal
		#endif
	}
	__device__ __inline__ cudaVec3<float> SampleHemisphere(
		const float r1,
		const float r2,
		const cudaVec3<float>& vN)
	{
		return SampleSphere(r1, r2 * 0.5f, vN);

		// ~~~~ fast approximation ~~~~
		//// calculate phi and theta angles
		//const float phi = r1 * 6.283185f;
		//const float theta = r2;

		//// calculate sample direction
		//#if defined(__CUDACC__)
		//const float sin_theta = sqrtf(theta);
		//return vX * sin_theta * __cosf(phi) + vY * sin_theta * __sinf(phi) + vN * sqrtf(1.0f - theta);
		////				  along local x axis		+ along local z axis		+ along normal
		//#endif
	}

	__device__ __inline__ void RandomVectorOnAngularSphere(
		const float& unsigned_random_theta,
		const float& usnigned_random_phi,
		const cudaVec3<float>& normal,
		cudaVec3<float>& randomVector)
	{
		// create local coordinate space vectors
		cudaVec3<float> ax, ay;
		if (fabs(normal.x) > fabs(normal.y))	ax = cudaVec3<float>(0.0f, 1.0f, 0.0f);
		else									ax = cudaVec3<float>(1.0f, 0.0f, 0.0f);

		ay = cudaVec3<float>::CrossProduct(normal, ax);
		ax = cudaVec3<float>::CrossProduct(normal, ay);


		// calculate phi and theta angles
		#if defined(__CUDACC__)
		float theta = unsigned_random_theta * 6.283185f;
		float phi = acosf(1.0f - 2.0f * usnigned_random_phi / 20.0f);

		// calculate sample directioN
		randomVector = ax * __sinf(phi) * __cosf(theta) + ay * __sinf(phi) * __sinf(theta) + normal * __cosf(phi);
		//				  along local x axis		+ along local z axis		+ along normal
		#endif
		//// calculate phi and theta angles
		//const float theta = unsigned_random_theta * 6.283185f;
		////const float phi = acosf(signed_random_phi);
		//const float phi = 0.5f * acosf(1.0f - 2.0f * signed_random_phi);
		////float theta = 0.5f * acosf(1.0f - 2.0f * r2);

		//// calculate sample direction
		//const float sin_phi = sinf(phi);
		//randomVector = ax * sinf(theta) * sin_phi + ay * cosf(theta) * sin_phi + normal * cosf(phi);
		////				  along local x axis		+ along local z axis		+ along normal
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}

#endif // !CUDA_RENDER_PARTS_CUH