#ifndef CUDA_RENDER_PARTS_CUH
#define CUDA_RENDER_PARTS_CUH

#include "cuda_engine_parts.cuh"
#include "world_object.h"
#include "vec3.h"
#include "vec2.h"
#include "color.h"

#include "render_parts.h"

#include "math_constants.h"

namespace RayZath::Cuda
{
	struct __align__(16u) vec3f
	{
	public:
		float x, y, z;

	public:
		__host__ __device__ explicit constexpr vec3f(const float value = 0.0f) noexcept
			: x(value)
			, y(value)
			, z(value)
		{}
		__host__ __device__ constexpr vec3f(const vec3f & V)
			: x(V.x)
			, y(V.y)
			, z(V.z)
		{}
		__host__ __device__ constexpr vec3f(const float x, const float y, const float z) noexcept
			: x(x)
			, y(y)
			, z(z)
		{}
		__host__ explicit constexpr vec3f(const Math::vec3<float> &v)
			: x(v.x)
			, y(v.y)
			, z(v.z)
		{}


	public:
		__device__ constexpr static float DotProduct(const vec3f & V1, const vec3f & V2) noexcept
		{
			return V1.x * V2.x + V1.y * V2.y + V1.z * V2.z;
		}
		__device__ static vec3f CrossProduct(const vec3f & V1, const vec3f & V2) noexcept
		{
			return vec3f(
				V1.y * V2.z - V1.z * V2.y,
				V1.z * V2.x - V1.x * V2.z,
				V1.x * V2.y - V1.y * V2.x);
		}
		__device__ static float Similarity(const vec3f & V1, const vec3f & V2)
		{
			return DotProduct(V1, V2) * (V1.RcpLength() * V2.RcpLength());
		}
		__device__ static float Distance(const vec3f & V1, const vec3f & V2)
		{
			return (V1 - V2).Length();
		}
		__device__ static vec3f Normalize(const vec3f & V)
		{
			vec3f normalized = V;
			normalized.Normalize();
			return normalized;
		}
		__device__ static vec3f Reverse(const vec3f & V) noexcept
		{
			return vec3f(-V.x, -V.y, -V.z);
		}


	public:
		__device__ constexpr float DotProduct(const vec3f & V) const noexcept
		{
			return (x * V.x + y * V.y + z * V.z);
		}
		__device__ constexpr void CrossProduct(const vec3f & V)
		{
			this->x = this->y * V.z - this->z * V.y;
			this->y = this->z * V.x - this->x * V.z;
			this->z = this->x * V.y - this->y * V.x;
		}
		__device__ float Similarity(const vec3f & V)
		{
			return this->DotProduct(V) * (this->RcpLength() * V.RcpLength());
		}
		__device__ float Distance(const vec3f & V)
		{
			return (*this - V).Length();
		}
		__device__ void Normalize()
		{
			const float scalar = RcpLength();
			x *= scalar;
			y *= scalar;
			z *= scalar;
		}
		__device__ vec3f Normalized() const
		{
			return Normalize(*this);
		}
		__device__ constexpr void Reverse()
		{
			x = -x;
			y = -y;
			z = -z;
		}
		__device__ constexpr vec3f Reversed() const
		{
			return -(*this);
		}
		__device__ void RotateX(const float angle)
		{
			#ifdef __CUDACC__
			float sina, cosa;
			cui_sincosf(angle, &sina, &cosa);
			float newY = y * cosa + z * sina;
			z = y * -sina + z * cosa;
			y = newY;
			#endif
		}
		__device__ void RotateY(const float angle)
		{
			#ifdef __CUDACC__
			float sina, cosa;
			cui_sincosf(angle, &sina, &cosa);
			float newX = x * cosa - z * sina;
			z = x * sina + z * cosa;
			x = newX;
			#endif
		}
		__device__ void RotateZ(const float angle)
		{
			#ifdef __CUDACC__
			float sina, cosa;
			cui_sincosf(angle, &sina, &cosa);
			float newX = x * cosa + y * sina;
			y = x * -sina + y * cosa;
			x = newX;
			#endif
		}

		__device__ void RotateXYZ(const vec3f & rot)
		{
			#ifdef __CUDACC__
			// x rotation
			float sina, cosa, newValue;
			cui_sincosf(rot.x, &sina, &cosa);
			newValue = y * cosa + z * sina;	// new y
			z = y * -sina + z * cosa;		// new z
			y = newValue;

			// y rotation
			cui_sincosf(rot.y, &sina, &cosa);
			newValue = x * cosa - z * sina;	// new x
			z = x * sina + z * cosa;		// new z
			x = newValue;

			// z rotation
			cui_sincosf(rot.z, &sina, &cosa);
			newValue = x * cosa + y * sina;	// new x
			y = x * -sina + y * cosa;		// new y
			x = newValue;
			#endif
		}
		__device__ void RotateZYX(const vec3f & rot)
		{
			#ifdef __CUDACC__
			// z rotation
			float sina, cosa, newValue;
			cui_sincosf(rot.z, &sina, &cosa);
			newValue = x * cosa + y * sina;
			y = x * -sina + y * cosa;
			x = newValue;

			// y rotation
			cui_sincosf(rot.y, &sina, &cosa);
			newValue = x * cosa - z * sina;
			z = x * sina + z * cosa;
			x = newValue;

			// x rotation
			cui_sincosf(rot.x, &sina, &cosa);
			newValue = y * cosa + z * sina;
			z = y * -sina + z * cosa;
			y = newValue;
			#endif
		}


	public:
		__host__ __device__ constexpr vec3f operator-() const noexcept
		{
			return vec3f(-x, -y, -z);
		}
		__host__ __device__ constexpr vec3f operator+(const vec3f & V) const noexcept
		{
			return vec3f(x + V.x, y + V.y, z + V.z);
		}
		__host__ __device__ constexpr vec3f operator-(const vec3f & V) const noexcept
		{
			return vec3f(x - V.x, y - V.y, z - V.z);
		}
		__host__ __device__ constexpr vec3f operator*(const float scalar) const noexcept
		{
			return vec3f(x * scalar, y * scalar, z * scalar);
		}
		__host__ __device__ constexpr vec3f operator*(const vec3f & scalar) const noexcept
		{
			return vec3f(x * scalar.x, y * scalar.y, z * scalar.z);
		}
		__host__ __device__ constexpr vec3f operator/(const float scalar) const
		{
			return vec3f(x / scalar, y / scalar, z / scalar);
		}
		__host__ __device__ constexpr vec3f operator/(const vec3f & scalar) const
		{
			return vec3f(x / scalar.x, y / scalar.y, z / scalar.z);
		}
		__host__ __device__ constexpr vec3f& operator+=(const vec3f & V)
		{
			x += V.x;
			y += V.y;
			z += V.z;
			return *this;
		}
		__host__ __device__ constexpr vec3f& operator-=(const vec3f & V)
		{
			this->x -= V.x;
			this->y -= V.y;
			this->z -= V.z;
			return *this;
		}
		__host__ __device__ constexpr vec3f& operator*=(const float scalar)
		{
			this->x *= scalar;
			this->y *= scalar;
			this->z *= scalar;
			return *this;
		}
		__host__ __device__ constexpr vec3f& operator*=(const vec3f & scalar)
		{
			this->x *= scalar.x;
			this->y *= scalar.y;
			this->z *= scalar.z;
			return *this;
		}
		__host__ __device__ constexpr vec3f& operator/=(const float scalar)
		{
			float rcp = 1.0f / scalar;
			this->x *= rcp;
			this->y *= rcp;
			this->z *= rcp;
			return *this;
		}
		__host__ __device__ constexpr vec3f& operator/=(const vec3f & scalar)
		{
			this->x /= scalar.x;
			this->y /= scalar.y;
			this->z /= scalar.z;
			return *this;
		}
		__host__ __device__ constexpr vec3f& operator=(const vec3f & V)
		{
			x = V.x;
			y = V.y;
			z = V.z;
			return *this;
		}
		__host__ constexpr vec3f& operator=(const Math::vec3<float>&v)
		{
			x = v.x;
			y = v.y;
			z = v.z;
			return *this;
		}


	public:
		__device__ constexpr void SetValues(const float xx, const float yy, const float zz)
		{
			this->x = xx;
			this->y = yy;
			this->z = zz;
		}
		__device__ float Length() const
		{
			return sqrtf(x * x + y * y + z * z);
		}
		__device__ constexpr float LengthSquared() const noexcept
		{
			return x * x + y * y + z * z;
		}
		__device__ float RcpLength() const noexcept
		{
			#ifdef __CUDACC__
			return rnorm3df(x, y, z);
			#else
			return 1.0f / Length();
			#endif
		}
	};
	template <typename T>
	struct vec2
	{
	public:
		T x, y;

	public:
		__host__ __device__ explicit constexpr vec2(const T value = T()) noexcept
			: x(value)
			, y(value)
		{}
		template <typename U>
		__host__ __device__ explicit constexpr vec2(const vec2<U> v)
			: x(T(v.x))
			, y(T(v.y))
		{}
		__host__ __device__ constexpr vec2(const T x, const T y)
			: x(x)
			, y(y)
		{}
		__host__ explicit constexpr vec2(const Math::vec2<T> v)
			: x(v.x)
			, y(v.y)
		{}

	public:
		__device__ constexpr static T DotProduct(const vec2 v1, const vec2 v2)
		{
			return v1.x * v2.x + v1.y * v2.y;
		}
		__device__ static T Similarity(const vec2 v1, const vec2 v2)
		{
			return DotProduct(v1, v2) * (v1.RcpLength() * v2.RcpLength());
		}
		__device__ static T Distance(const vec2 v1, const vec2 v2)
		{
			return (v1 - v2).Length();
		}
		__device__ static vec2 Normalize(const vec2 v)
		{
			v.Normalized();
		}
		__device__ static vec2 Reverse(const vec2 v) noexcept
		{
			return vec2(-v.x, -v.y, -v.z);
		}

	public:
		__device__ constexpr T DotProduct(const vec2 v) const noexcept
		{
			return (x * v.x + y * v.y);
		}
		__device__ T Similarity(const vec2 v)
		{
			return this->DotProduct(v) * (this->RcpLength() * v.RcpLength());
		}
		__device__ T Distance(const vec2 v)
		{
			return (*this - v).Length();
		}
		__device__ void Normalize()
		{
			const T scalar = RcpLength();
			x *= scalar;
			y *= scalar;
		}
		__device__ vec2 Normalized() const
		{
			return Normalize(*this);
		}
		__device__ constexpr void Reverse()
		{
			x = -x;
			y = -y;
		}
		__device__ constexpr vec2 Reversed() const
		{
			return -(*this);
		}
		__device__ void Rotate(const float angle)
		{
			#ifndef __CUDACC__
			float sina, cosa;
			cui_sincosf(angle, &sina, &cosa);
			const float xx = x * cosa - y * sina;
			const float yy = x * sina + y * cosa;
			x = xx;
			y = yy;
			#endif
		}
		__device__ vec2 Rotated(const float angle) const
		{
			vec2 v = *this;
			v.Rotate(angle);
			return v;
		}

	public:
		__host__ __device__ constexpr vec2 operator-() const noexcept
		{
			return vec2(-x, -y);
		}
		__host__ __device__ constexpr vec2 operator+(const vec2 v) const noexcept
		{
			return vec2(x + v.x, y + v.y);
		}
		__host__ __device__ constexpr vec2 operator-(const vec2 v) const noexcept
		{
			return vec2(x - v.x, y - v.y);
		}
		__host__ __device__ constexpr vec2 operator*(const T scalar) const noexcept
		{
			return vec2(x * scalar, y * scalar);
		}
		__host__ __device__ constexpr vec2 operator*(const vec2 scalar) const noexcept
		{
			return vec2(x * scalar.x, y * scalar.y);
		}
		__host__ __device__ constexpr vec2 operator/(const T scalar) const
		{
			return vec2(x / scalar, y / scalar);
		}
		__host__ __device__ constexpr vec2 operator/(const vec2 scalar) const
		{
			return vec2(x / scalar.x, y / scalar.y);
		}
		__host__ __device__ constexpr vec2& operator+=(const vec2 v)
		{
			x += v.x;
			y += v.y;
			return *this;
		}
		__host__ __device__ constexpr vec2& operator-=(const vec2 v)
		{
			x -= v.x;
			y -= v.y;
			return *this;
		}
		__host__ __device__ constexpr vec2& operator*=(const T scalar)
		{
			x *= scalar;
			y *= scalar;
			return *this;
		}
		__host__ __device__ constexpr vec2& operator*=(const vec2 scalar)
		{
			x *= scalar.x;
			y *= scalar.y;
			return *this;
		}
		__host__ __device__ constexpr vec2& operator/=(const T scalar)
		{
			x /= scalar;
			y /= scalar;
			return *this;
		}
		__host__ __device__ constexpr vec2& operator/=(const vec2 scalar)
		{
			x /= scalar.x;
			y /= scalar.y;
			return *this;
		}
		__host__ __device__ constexpr vec2& operator=(const vec2 v)
		{
			x = v.x;
			y = v.y;
			return *this;
		}
		__host__ constexpr vec2& operator=(const Math::vec2<T> v)
		{
			x = v.x;
			y = v.y;
			return *this;
		}
		__host__ constexpr bool operator==(const Math::vec2<T> v) const
		{
			return x == v.x && y == v.y;
		}
		__host__ constexpr bool operator!=(const Math::vec2<T> v) const
		{
			return !(*this == v);
		}

	public:
		__device__ T Length() const
		{
			return sqrt(x * x + y * y);
		}
		__device__ T LengthSquared() const
		{
			return x * x + y * y;
		}
		__device__ T RcpLength() const
		{
			return 1.0f / Length();
		}
	};
	typedef vec2<float> vec2f;
	typedef vec2<uint32_t> vec2ui32;
	typedef vec2<uint16_t> vec2ui16;

	template <typename T> class Color;
	template<>
	class Color<float>
	{
	public:
		float red, green, blue, alpha;


	public:
		__host__ __device__ constexpr Color()
			: red(1.0f)
			, green(1.0f)
			, blue(1.0f)
			, alpha(1.0f)
		{}
		__host__ __device__ constexpr Color(const Color<float>& color)
			: red(color.red)
			, green(color.green)
			, blue(color.blue)
			, alpha(color.alpha)
		{}
		template <typename U>
		__host__ __device__ constexpr Color(const Color<U>& color)
			: red(float(color.red))
			, green(float(color.green))
			, blue(float(color.blue))
			, alpha(float(color.alpha))
		{}
		__host__ __device__ explicit constexpr Color(const float value)
			: red(value)
			, green(value)
			, blue(value)
			, alpha(value)
		{}
		__host__ __device__ constexpr Color(
			const float red,
			const float green,
			const float blue,
			const float alpha)
			: red(red)
			, green(green)
			, blue(blue)
			, alpha(alpha)
		{}
		__host__ Color(const Graphics::Color& color)
			: red(color.red / 255.0f)
			, green(color.green / 255.0f)
			, blue(color.blue / 255.0f)
			, alpha(color.alpha / 255.0f)
		{}


	public:
		__device__ constexpr Color<float> operator*(const float factor) const
		{
			return Color<float>(
				this->red * factor,
				this->green * factor,
				this->blue * factor,
				this->alpha * factor);
		}
		__device__ constexpr Color<float> operator*(const Color<float>& other) const
		{
			return Color<float>(
				red * other.red,
				green * other.green,
				blue * other.blue,
				alpha * other.alpha);
		}
		__device__ constexpr Color<float> operator+(const Color<float>& color) const
		{
			return Color<float>(
				this->red + color.red,
				this->green + color.green,
				this->blue + color.blue,
				this->alpha + color.alpha);
		}
		__device__ constexpr Color<float> operator-(const Color<float>& color) const
		{
			return Color<float>(
				this->red - color.red,
				this->green - color.green,
				this->blue - color.blue,
				this->alpha - color.alpha);
		}
		__device__ constexpr Color<float> operator/(float factor) const
		{
			factor = 1.0f / factor;
			return Color<float>(
				this->red * factor,
				this->green * factor,
				this->blue * factor,
				this->alpha * factor);
		}
		__device__ constexpr Color<float> operator/(const Color<float>& divisor) const
		{
			return Color<float>(
				this->red / divisor.red,
				this->green / divisor.green,
				this->blue / divisor.blue,
				this->alpha / divisor.alpha);
		}
		__host__ __device__ constexpr Color<float>& operator=(const Color<float>& color)
		{
			this->red = color.red;
			this->green = color.green;
			this->blue = color.blue;
			this->alpha = color.alpha;
			return *this;
		}
		__host__ Color<float>& operator=(const Graphics::Color& color)
		{
			this->red = color.red / 255.0f;
			this->green = color.green / 255.0f;
			this->blue = color.blue / 255.0f;
			this->alpha = color.alpha / 255.0f;
			return *this;
		}
		__device__ constexpr Color<float>& operator*=(const float factor)
		{
			this->red *= factor;
			this->green *= factor;
			this->blue *= factor;
			this->alpha *= factor;
			return *this;
		}
		__device__ constexpr Color<float>& operator*=(const Color<float>& other)
		{
			red *= other.red;
			green *= other.green;
			blue *= other.blue;
			alpha *= other.alpha;
			return *this;
		}
		__device__ constexpr Color<float>& operator+=(const Color<float>& color)
		{
			this->red += color.red;
			this->green += color.green;
			this->blue += color.blue;
			this->alpha += color.alpha;
			return *this;
		}
		__device__ constexpr Color<float>& operator/=(float factor)
		{
			factor = 1.0f / factor;
			this->red *= factor;
			this->green *= factor;
			this->blue *= factor;
			this->alpha *= factor;
			return *this;
		}
		__device__ constexpr Color<float>& operator/=(const Color<float>& factor)
		{
			red /= factor.red;
			green /= factor.green;
			blue /= factor.blue;
			alpha /= factor.alpha;
			return *this;
		}


	public:
		__device__ static constexpr Color<float> Blend(
			const Color<float>& color1,
			const Color<float>& color2,
			float t)
		{
			return color1 + (color2 - color1) * t;
		}
		__device__ constexpr void Blend(const Color<float>& color, const float t)
		{
			*this = Blend(*this, color, t);
		}

		__host__ __device__ constexpr void Set(
			const float r,
			const float g,
			const float b,
			const float a)
		{
			red = r;
			green = g;
			blue = b;
			alpha = a;
		}
	};
	typedef Color<float> ColorF;

	template<>
	class Color<unsigned char>
	{
	public:
		unsigned char red, green, blue, alpha;


	public:
		__device__ constexpr Color()
			: red(255)
			, green(255)
			, blue(255)
			, alpha(255)
		{}
		__device__ constexpr Color(
			const unsigned char red,
			const unsigned char green,
			const unsigned char blue,
			const unsigned char alpha = 0xFF)
			: red(red)
			, green(green)
			, blue(blue)
			, alpha(alpha)
		{}
		__host__ constexpr Color(const Color<unsigned char>& color)
			: red(color.red)
			, green(color.green)
			, blue(color.blue)
			, alpha(color.alpha)
		{}
		__device__ constexpr Color(const Color<float>& color)
			: red(unsigned char(color.red * 255.0f))
			, green(unsigned char(color.green * 255.0f))
			, blue(unsigned char(color.blue * 255.0f))
			, alpha(unsigned char(color.alpha * 255.0f))
		{}
		__host__ Color(const Graphics::Color& color)
			: red(color.red)
			, green(color.green)
			, blue(color.blue)
			, alpha(color.alpha)
		{}


	public:
		__host__ __device__ constexpr Color<unsigned char>& operator=(const Color<unsigned char>& color)
		{
			this->red = color.red;
			this->green = color.green;
			this->blue = color.blue;
			this->alpha = color.alpha;
			return *this;
		}
		__host__ Color<unsigned char>& operator=(const Graphics::Color& color)
		{
			this->red = color.red;
			this->green = color.green;
			this->blue = color.blue;
			this->alpha = color.alpha;
			return *this;
		}
		__device__ constexpr Color<unsigned char>& operator*=(const float factor)
		{
			this->red = static_cast<unsigned char>(this->red * factor);
			this->green = static_cast<unsigned char>(this->green * factor);
			this->blue = static_cast<unsigned char>(this->blue * factor);
			this->alpha = static_cast<unsigned char>(this->alpha * factor);
			return *this;
		}
		__device__ constexpr Color<unsigned char> operator*(const float factor) const
		{
			return Color<unsigned char>(
				static_cast<unsigned char>(this->red * factor),
				static_cast<unsigned char>(this->green * factor),
				static_cast<unsigned char>(this->blue * factor),
				static_cast<unsigned char>(this->alpha * factor));
		}


	public:
		__device__ static Color<unsigned char> Blend(const Color<unsigned char>& color1, const Color<unsigned char>& color2)
		{
			return Color<unsigned char>(
				(color1.red + color2.red) / 2,
				(color1.green + color2.green) / 2,
				(color1.blue + color2.blue) / 2,
				(color1.alpha + color2.alpha) / 2);
		}
		__device__ static Color<unsigned char> Blend(const Color<unsigned char>& color1, const Color<unsigned char>& color2, const unsigned char balance)
		{
			return Color<unsigned char>(
				(color1.red * balance + color2.red * (255u - balance)) / 255u,
				(color1.green * balance + color2.green * (255u - balance)) / 255u,
				(color1.blue * balance + color2.blue * (255u - balance)) / 255u,
				(color1.alpha * balance + color2.blue * (255u - balance)) / 255u);
		}
		__device__ static Color<unsigned char> BlendProduct(const Color<unsigned char>& color1, const Color<unsigned char>& color2)
		{
			return Color<unsigned char>(
				(color1.red * color2.red) / 255u,
				(color1.green * color2.green) / 255u,
				(color1.blue * color2.blue) / 255u,
				(color1.alpha * color2.alpha) / 255u);
		}


	public:
		__device__ void Blend(const Color<unsigned char>& color)
		{
			this->red = (this->red + color.red) / 2;
			this->green = (this->green + color.green) / 2;
			this->blue = (this->blue + color.blue) / 2;
			this->alpha = (this->alpha + color.alpha) / 2;
		}
		__device__ void Blend(const Color<unsigned char>& color, const unsigned char balance)
		{
			this->red = (this->red * (255u - balance) + color.red * balance) / 255u;
			this->green = (this->green * (255u - balance) + color.green * balance) / 255u;
			this->blue = (this->blue * (255u - balance) + color.blue * balance) / 255u;
			this->alpha = (this->alpha * (255u - balance) + color.alpha * balance) / 255u;
		}
		__device__ void BlendProduct(const Color<unsigned char>& color)
		{
			*this = Color<unsigned char>::BlendProduct(*this, color);
		}


	public:
		__device__ void SetColor(
			const unsigned char r,
			const unsigned char g,
			const unsigned char b,
			const unsigned char a = 0xFF)
		{
			red = r;
			green = g;
			blue = b;
			alpha = a;
		}
	};
	typedef Color<unsigned char> ColorU;

	class World;
	struct Material;
	typedef vec2f Texcrd;

	struct BlockThread
	{
		uint32_t block_pos;
		uint32_t block_idx;

		__device__ __inline__ BlockThread()
			: block_pos(threadIdx.y* blockDim.x + threadIdx.x)
			, block_idx(blockIdx.y* gridDim.x + blockIdx.x)
		{}
	};
	struct GridThread
	{
		vec2ui32 grid_pos;
		uint32_t grid_idx;

		__device__ __inline__ GridThread()
			: grid_pos(blockIdx.x* blockDim.x + threadIdx.x, blockIdx.y* blockDim.y + threadIdx.y)
			, grid_idx(
				(blockIdx.y* gridDim.x + blockIdx.x)*
				(blockDim.x* blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x)
		{}
	};
	struct FullThread
		: public BlockThread
		, public GridThread
	{};

	struct RNG
	{
	private:
		float a, b;

	public:
		__device__ RNG(const vec2f seed, const float r)
			: a(seed.x + seed.y)
			, b(r * 245.310913f)
		{}
		__device__ float UnsignedUniform()
		{
			a = fract((a + 0.2311362f) * (b + 13.054377f));
			b = fract((a + 251.78431f) + (b - 73.054312f));
			return fabsf(b);
		}
		__device__ float SignedUniform()
		{
			return UnsignedUniform() * 2.0f - 1.0f;
		}
	private:
		__device__ float fract(const float f)
		{
			return f - truncf(f);
		}
	};


	struct Ray
	{
	public:
		vec3f origin;
		vec3f direction;

	public:
		Ray() = default;
		__device__ Ray(const vec3f origin, const vec3f direction)
			: origin(origin)
			, direction(direction)
		{
			this->direction.Normalize();
		}
	};
	struct RangedRay : public Ray
	{
	public:
		vec2f near_far;

	public:
		__device__ RangedRay()
			: near_far(0.0f, 3.402823466e+38f)
		{}
		__device__ RangedRay(
			const vec3f& origin,
			const vec3f& direction,
			const vec2f near_far = vec2f(0.0f, 3.402823466e+38f))
			: Ray(origin, direction)
			, near_far(near_far)
		{}

	public:
		__device__ void ResetRange(const vec2f range = vec2f(0.0f, 3.402823466e+38f))
		{
			near_far = range;
		}
	};
	struct SceneRay : public RangedRay
	{
	public:
		const Material* material;
		ColorF color;


	public:
		__device__ SceneRay()
			: RangedRay()
			, material(nullptr)
			, color(1.0f)
		{}
		__device__ SceneRay(
			const vec3f& origin,
			const vec3f& direction,
			const Material* material,
			const ColorF& color = ColorF(1.0f),
			const vec2f near_far = vec2f(0.0f, 3.402823466e+38f))
			: RangedRay(origin, direction, near_far)
			, material(material)
			, color(color)
		{}
	};

	class Mesh;
	struct Triangle;
	struct TraversalResult
	{
		const Mesh* closest_object = nullptr;
		const Triangle* closest_triangle = nullptr;
		vec2f barycenter;
		bool external = true;
	};
	struct SurfaceProperties
	{
		const Material* surface_material;
		const Material* behind_material;

		Texcrd texcrd;
		vec3f normal;
		vec3f mapped_normal;

		ColorF color;
		float metalness = 0.0f;
		float roughness = 0.0f;
		float emission = 0.0f;

		float fresnel = 1.0f;
		float reflectance = 0.0f;

		float tint_factor = 0.0f;
		vec2f refraction_factors;

		__device__ SurfaceProperties(const Material* material)
			: surface_material(material)
			, behind_material(material)
		{}
	};
	struct TracingResult
	{
		vec3f point;
		vec3f next_direction;

		__device__ void RepositionRay(SceneRay& ray) const
		{
			ray.origin = point;
			ray.direction = next_direction;
			ray.ResetRange();
		}
	};

	struct Triangle
	{
	private:
		vec3f m_v1, m_v2, m_v3;
		vec3f m_n1, m_n2, m_n3;
		vec3f m_normal;
		Texcrd m_t1, m_t2, m_t3;
		uint32_t m_material_id;

	public:
		__host__ Triangle(const RayZath::Engine::Triangle& hostTriangle);

	public:
		__host__ void SetVertices(const vec3f& v1, const vec3f& v2, const vec3f& v3);
		__host__ void SetTexcrds(const vec2f& t1, const vec2f& t2, const vec2f& t3);
		__host__ void SetNormals(const vec3f& n1, const vec3f& n2, const vec3f& n3);

	public:
		__device__ uint32_t GetMaterialId() const
		{
			return m_material_id;
		}
		__device__ const vec3f& GetNormal() const
		{
			return m_normal;
		}

		__device__ bool ClosestIntersection(RangedRay& ray, TraversalResult& traversal) const
		{
			const vec3f edge1 = m_v2 - m_v1;
			const vec3f edge2 = m_v3 - m_v1;
			const vec3f pvec = vec3f::CrossProduct(ray.direction, edge2);

			float det = (vec3f::DotProduct(edge1, pvec));
			det += static_cast<float>(det > -1.0e-7f && det < 1.0e-7f) * 1.0e-7f;
			const float inv_det = 1.0f / det;

			const vec3f tvec = ray.origin - m_v1;
			const float b1 = vec3f::DotProduct(tvec, pvec) * inv_det;
			if (b1 < 0.0f || b1 > 1.0f)
				return false;

			const vec3f qvec = vec3f::CrossProduct(tvec, edge1);

			const float b2 = vec3f::DotProduct(ray.direction, qvec) * inv_det;
			if (b2 < 0.0f || b1 + b2 > 1.0f)
				return false;

			const float t = vec3f::DotProduct(edge2, qvec) * inv_det;
			if (t <= ray.near_far.x || t >= ray.near_far.y)
				return false;

			ray.near_far.y = t;
			traversal.closest_triangle = this;
			traversal.external = det > 0.0f;
			traversal.barycenter = vec2f(b1, b2);

			return true;
		}
		__device__ bool AnyIntersection(RangedRay& ray, vec2f& barycenter) const
		{
			const vec3f edge1 = m_v2 - m_v1;
			const vec3f edge2 = m_v3 - m_v1;
			const vec3f pvec = vec3f::CrossProduct(ray.direction, edge2);

			float det = (vec3f::DotProduct(edge1, pvec));
			det += static_cast<float>(det > -1.0e-7f && det < 1.0e-7f) * 1.0e-7f;
			const float inv_det = 1.0f / det;

			const vec3f tvec = ray.origin - m_v1;
			const float b1 = vec3f::DotProduct(tvec, pvec) * inv_det;
			if (b1 < 0.0f || b1 > 1.0f)
				return false;

			const vec3f qvec = vec3f::CrossProduct(tvec, edge1);

			const float b2 = vec3f::DotProduct(ray.direction, qvec) * inv_det;
			if (b2 < 0.0f || b1 + b2 > 1.0f)
				return false;

			const float t = vec3f::DotProduct(edge2, qvec) * inv_det;
			if (t <= ray.near_far.x || t >= ray.near_far.y)
				return false;

			barycenter = vec2f(b1, b2);

			return true;
		}
		__device__ Texcrd TexcrdFromBarycenter(const vec2f barycenter) const
		{
			const float b3 = 1.0f - barycenter.x - barycenter.y;
			const float u = m_t1.x * b3 + m_t2.x * barycenter.x + m_t3.x * barycenter.y;
			const float v = m_t1.y * b3 + m_t2.y * barycenter.x + m_t3.y * barycenter.y;
			return Texcrd(u, v);
		}
		__device__ vec3f AverageNormal(const vec2f barycenter) const
		{
			return  (m_n1 * (1.0f - barycenter.x - barycenter.y) + m_n2 * barycenter.x + m_n3 * barycenter.y).Normalized();
		}
		__device__ void MapNormal(const ColorF& map_color, vec3f& mapped_normal) const
		{
			const vec3f edge1 = m_v2 - m_v1;
			const vec3f edge2 = m_v3 - m_v1;
			const vec2f dUV1 = m_t2 - m_t1;
			const vec2f dUV2 = m_t3 - m_t1;

			// tangent and bitangent
			const float f = 1.0f / (dUV1.x * dUV2.y - dUV2.x * dUV1.y);
			vec3f tangent = ((edge1 * dUV2.y - edge2 * dUV1.y) * f).Normalized();
			// tangent re-orthogonalization
			tangent = (tangent - mapped_normal * vec3f::DotProduct(tangent, mapped_normal)).Normalized();
			// bitangent is simply cross product of normal and tangent
			vec3f bitangent = vec3f::CrossProduct(tangent, mapped_normal);

			// map normal transformation to [-1.0f, 1.0f] range
			const vec3f map_normal = vec3f(map_color.red, map_color.green, map_color.blue) * 2.0f - vec3f(1.0f);

			// calculate normal
			mapped_normal = mapped_normal * map_normal.z + tangent * map_normal.x + bitangent * map_normal.y;
		}
	};


	struct CoordSystem
	{
	public:
		vec3f x_axis, y_axis, z_axis;

	public:
		__host__ CoordSystem(
			const Math::vec3f& x = Math::vec3f(1.0f, 0.0f, 0.0f),
			const Math::vec3f& y = Math::vec3f(0.0f, 1.0f, 0.0f),
			const Math::vec3f& z = Math::vec3f(0.0f, 0.0f, 1.0f))
			: x_axis(x)
			, y_axis(y)
			, z_axis(z)
		{}

	public:
		__host__ CoordSystem& operator=(const RayZath::Engine::CoordSystem& coordSystem)
		{
			x_axis = coordSystem.GetXAxis();
			y_axis = coordSystem.GetYAxis();
			z_axis = coordSystem.GetZAxis();
			return *this;
		}


	public:
		__device__ void TransformBackward(vec3f& v) const
		{
			v = x_axis * v.x + y_axis * v.y + z_axis * v.z;
		}
		__device__ void TransformForward(vec3f& v) const
		{
			v = vec3f(
				x_axis.x * v.x + x_axis.y * v.y + x_axis.z * v.z,
				y_axis.x * v.x + y_axis.y * v.y + y_axis.z * v.z,
				z_axis.x * v.x + z_axis.y * v.y + z_axis.z * v.z);
		}
	};
	struct Transformation
	{
	private:
		vec3f position, scale;
		CoordSystem coord_system;

	public:
		__host__ Transformation& operator=(const RayZath::Engine::Transformation& t)
		{
			position = t.GetPosition();
			scale = t.GetScale();
			coord_system = t.GetCoordSystem();
			return *this;
		}

	public:
		__device__ __inline__ void TransformG2L(RangedRay& ray) const
		{
			ray.origin -= position;
			coord_system.TransformForward(ray.origin);
			ray.origin /= scale;

			coord_system.TransformForward(ray.direction);
			ray.direction /= scale;
		}
		__device__ __inline__ void TransformL2G(vec3f& v) const
		{
			v /= scale;
			coord_system.TransformBackward(v);
		}
	};

	struct BoundingBox
	{
		vec3f min, max;

		__host__ BoundingBox()
			: min(0.0f)
			, max(0.0f)
		{}
		__host__ BoundingBox(const RayZath::Engine::BoundingBox& box)
			: min(box.min)
			, max(box.max)
		{}

		__host__ BoundingBox& operator=(const RayZath::Engine::BoundingBox& box)
		{
			this->min = box.min;
			this->max = box.max;
			return *this;
		}

		__device__ __inline__ bool RayIntersection(const RangedRay& ray) const
		{
			float t1 = (min.x - ray.origin.x) / ray.direction.x;
			float t2 = (max.x - ray.origin.x) / ray.direction.x;
			float t3 = (min.y - ray.origin.y) / ray.direction.y;
			float t4 = (max.y - ray.origin.y) / ray.direction.y;
			float t5 = (min.z - ray.origin.z) / ray.direction.z;
			float t6 = (max.z - ray.origin.z) / ray.direction.z;

			float tmin = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
			float tmax = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));

			return !(tmax < ray.near_far.x || tmin > tmax || tmin > ray.near_far.y);
		}
	};


	// ~~~~~~~~ Helper Functions ~~~~~~~~
	template <typename T>
	__device__ __inline__ T Lerp(const T a, const T b, const float t)
	{
		return a + (b - a) * t;
	}

	__device__ __inline__ vec3f ReflectVector(
		const vec3f& vI,
		const vec3f& vN)
	{
		return (vN * -2.0f * vec3f::DotProduct(vN, vI) + vI);
	}
	__device__ __inline__ vec3f HalfwayVector(
		const vec3f& vI,
		const vec3f& vR)
	{
		return ((-vI) + vR).Normalized();
	}
	__device__ __inline__ float RayToPointDistance(
		const Ray& ray,
		const vec3f& P)
	{
		// O - ray origin
		// P - specified point
		// vD - ray direction

		const vec3f vOP = P - ray.origin;
		const float dOP = vOP.Length();
		const float vOP_dot_vD = vec3f::DotProduct(vOP, ray.direction);
		return sqrtf(dOP * dOP - vOP_dot_vD * vOP_dot_vD);
	}
	__device__ __inline__ void RayPointCalculation(
		const Ray& ray,
		const vec3f& P,
		vec3f& vOP,
		float& dOP,
		float& vOP_dot_vD,
		float& dPQ)
	{
		/*
			^
			|
			Q ---- P
			|     /
			|    /	    // O - ray origin
			|   /	    // P - specified point
			|  /	    // Q - closest point to P lying on ray
			| /
			|/
			O
		*/

		vOP = P - ray.origin;
		dOP = vOP.Length();
		vOP_dot_vD = vec3f::DotProduct(vOP, ray.direction);
		dPQ = sqrtf(dOP * dOP - vOP_dot_vD * vOP_dot_vD);
	}

	__device__ __inline__ void LocalCoordinate(
		const vec3f& vN,
		vec3f& vX,
		vec3f& vY)
	{
		bool b = (fabs(vN.x) > fabs(vN.y));
		vX.x = static_cast<float>(!b);
		vX.y = static_cast<float>(b);
		vX.z = 0.0f;

		vY = vec3f::CrossProduct(vN, vX);
		vX = vec3f::CrossProduct(vN, vY);
	}

	__device__ __inline__ vec3f CosineSampleHemisphere(
		const float r1,
		const float r2,
		const vec3f& vN)
	{
		// create local coordinate space vectors
		vec3f vX, vY;
		LocalCoordinate(vN, vX, vY);

		const float phi = r1 * 6.283185f;
		const float theta = r2;

		// calculate sample direction
		const float sqrt_theta = sqrtf(theta);
		return vX * sqrt_theta * cui_cosf(phi) + vY * sqrt_theta * cui_sinf(phi) + vN * sqrtf(1.0f - theta);
		//				  along local x axis		+ along local z axis		+ along normal
	}
	__device__ __inline__ vec3f SampleSphere(
		const float r1,
		const float r2,
		const vec3f& vN)
	{
		// create local coordinate space vectors
		vec3f vX, vY;
		LocalCoordinate(vN, vX, vY);

		// calculate phi and theta angles
		const float phi = r1 * 6.283185f;
		const float theta = acosf(1.0f - 2.0f * r2);

		// calculate sample direction
		const float sin_theta = cui_sinf(theta);
		return vX * sin_theta * cui_cosf(phi) + vY * sin_theta * cui_sinf(phi) + vN * cui_cosf(theta);
		//		along local x axis			+ along local y axis			+ along normal
	}
	__device__ __inline__ vec3f SampleHemisphere(
		const float r1,
		const float r2,
		const vec3f& vN)
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
	__device__ __inline__ vec3f SampleDisk(
		const vec3f& vN,
		const float radius,
		RNG& rng)
	{
		vec3f vX, vY;
		LocalCoordinate(vN, vX, vY);
		const float r1 = rng.UnsignedUniform() * 2.0f * CUDART_PI_F;
		const float r2 = rng.UnsignedUniform();
		return (vX * cui_sinf(r1) + vY * cui_cosf(r1)) * sqrtf(r2) * radius;
	}

	// returns probability of reflection
	__device__ __inline__ float FresnelSpecularRatio(
		const vec3f& vN,
		const vec3f& vI,
		const float n1,
		const float n2,
		vec2f& factors)
	{
		const float ratio = n1 / n2;
		const float cosi = fabsf(vec3f::DotProduct(vI, vN));
		const float sin2_t = ratio * ratio * (1.0f - cosi * cosi);
		if (sin2_t >= 1.0f) return 1.0f; // total 'internal' reflection

		const float cost = sqrtf(1.0f - sin2_t);
		const float Rp = ((n1 * cosi) - (n2 * cost)) / ((n1 * cosi) + (n2 * cost));
		const float Rs = ((n2 * cosi) - (n1 * cost)) / ((n2 * cosi) + (n1 * cost));

		factors = vec2f(ratio, ratio * cosi - cost);
		return (Rs * Rs + Rp * Rp) / 2.0f;
	}
	__device__ __inline__ float FresnelSpecularRatioSchlick(
		const vec3f& vN,
		const vec3f& vI,
		const float n1,
		const float n2)
	{
		const float ratio = (n1 - n2) / (n1 + n2);
		const float r0 = ratio * ratio;
		const float vN_dot_vI = fabsf(vec3f::DotProduct(vN, vI));
		return r0 + (1.0f - r0) * cui_powf(1.0f - vN_dot_vI, 5.0f);
	}
}

#endif // !CUDA_RENDER_PARTS_CUH