#ifndef CUDA_RENDER_PARTS_CUH
#define CUDA_RENDER_PARTS_CUH

#include "cuda_engine_parts.cuh"
#include "render_object.h"
#include "vec3.h"
#include "vec2.h"
#include "color.h"

#include "render_parts.h"
#include "exist_flag.cuh"

#include "math_constants.h"

namespace RayZath
{
	namespace CudaEngine
	{
		struct __align__(16u) vec3f
		{
		public:
			float x, y, z;

		public:
			__host__ __device__ constexpr vec3f(const float& value = 0.0f) noexcept
				: x(value)
				, y(value)
				, z(value)
			{}
			__host__ __device__ constexpr vec3f(const vec3f & V)
				: x(V.x)
				, y(V.y)
				, z(V.z)
			{}
			__host__ __device__ constexpr vec3f(const float& x, const float& y, const float& z) noexcept
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
			__device__ void RotateX(const float& angle)
			{
				float sina, cosa;
				cui_sincosf(angle, &sina, &cosa);
				float newY = y * cosa + z * sina;
				z = y * -sina + z * cosa;
				y = newY;
			}
			__device__ void RotateY(const float& angle)
			{
				float sina, cosa;
				cui_sincosf(angle, &sina, &cosa);
				float newX = x * cosa - z * sina;
				z = x * sina + z * cosa;
				x = newX;
			}
			__device__ void RotateZ(const float& angle)
			{
				float sina, cosa;
				cui_sincosf(angle, &sina, &cosa);
				float newX = x * cosa + y * sina;
				y = x * -sina + y * cosa;
				x = newX;
			}

			__device__ void RotateXYZ(const vec3f & rot)
			{
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
			}
			__device__ void RotateZYX(const vec3f & rot)
			{
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
			__host__ __device__ constexpr vec3f operator*(const float& scalar) const noexcept
			{
				return vec3f(x * scalar, y * scalar, z * scalar);
			}
			__host__ __device__ constexpr vec3f operator*(const vec3f & scalar) const noexcept
			{
				return vec3f(x * scalar.x, y * scalar.y, z * scalar.z);
			}
			__host__ __device__ constexpr vec3f operator/(const float& scalar) const
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
			__host__ __device__ constexpr vec3f& operator*=(const float& scalar)
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
			__host__ __device__ constexpr vec3f& operator/=(const float& scalar)
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
			__device__ constexpr void SetValues(const float& xx, const float& yy, const float& zz)
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
		struct __align__(8) vec2
		{
		public:
			T x, y;

		public:
			__host__ __device__ constexpr vec2(const T & value = T()) noexcept
				: x(value)
				, y(value)
			{}
			__host__ __device__ constexpr vec2(const vec2 & v)
				: x(v.x)
				, y(v.y)
			{}
			template <typename U>
			__host__ __device__ explicit constexpr vec2(const vec2<U>& v)
				: x(T(v.x))
				, y(T(v.y))
			{}
			__host__ __device__ constexpr vec2(const T & x, const T & y)
				: x(x)
				, y(y)
			{}
			__host__ explicit constexpr vec2(const Math::vec2<T>&v)
				: x(v.x)
				, y(v.y)
			{}

		public:
			__device__ constexpr static T DotProduct(const vec2 & v1, const vec2 & v2)
			{
				return v1.x * v2.x + v1.y * v2.y;
			}
			__device__ static T Similarity(const vec2 & v1, const vec2 & v2)
			{
				return DotProduct(v1, v2) * (v1.RcpLength() * v2.RcpLength());
			}
			__device__ static T Distance(const vec2 & v1, const vec2 & v2)
			{
				return (v1 - v2).Length();
			}
			__device__ static vec2 Normalize(const vec2 & v)
			{
				v.Normalized();
			}
			__device__ static vec2 Reverse(const vec2 & v) noexcept
			{
				return vec2(-v.x, -v.y, -v.z);
			}

		public:
			__device__ constexpr T DotProduct(const vec2 & v) const noexcept
			{
				return (x * v.x + y * v.y);
			}
			__device__ T Similarity(const vec2 & v)
			{
				return this->DotProduct(v) * (this->RcpLength() * v.RcpLength());
			}
			__device__ T Distance(const vec2 & v)
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
			__device__ void Rotate(const float& angle)
			{
				float sina, cosa;
				cui_sincosf(angle, &sina, &cosa);
				const float xx = x * cosa - y * sina;
				const float yy = x * sina + y * cosa;
				x = xx;
				y = yy;
			}
			__device__ vec2 Rotated(const float& angle) const
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
			__host__ __device__ constexpr vec2 operator+(const vec2 & v) const noexcept
			{
				return vec2(x + v.x, y + v.y);
			}
			__host__ __device__ constexpr vec2 operator-(const vec2 & v) const noexcept
			{
				return vec2(x - v.x, y - v.y);
			}
			__host__ __device__ constexpr vec2 operator*(const T & scalar) const noexcept
			{
				return vec2(x * scalar, y * scalar);
			}
			__host__ __device__ constexpr vec2 operator*(const vec2 & scalar) const noexcept
			{
				return vec2(x * scalar.x, y * scalar.y);
			}
			__host__ __device__ constexpr vec2 operator/(const T & scalar) const
			{
				return vec2(x / scalar, y / scalar);
			}
			__host__ __device__ constexpr vec2 operator/(const vec2 & scalar) const
			{
				return vec2(x / scalar.x, y / scalar.y);
			}
			__host__ __device__ constexpr vec2& operator+=(const vec2 & v)
			{
				x += v.x;
				y += v.y;
				return *this;
			}
			__host__ __device__ constexpr vec2& operator-=(const vec2 & v)
			{
				x -= v.x;
				y -= v.y;
				return *this;
			}
			__host__ __device__ constexpr vec2& operator*=(const T & scalar)
			{
				x *= scalar;
				y *= scalar;
				return *this;
			}
			__host__ __device__ constexpr vec2& operator*=(const vec2 & scalar)
			{
				x *= scalar.x;
				y *= scalar.y;
				return *this;
			}
			__host__ __device__ constexpr vec2& operator/=(const T & scalar)
			{
				x /= scalar;
				y /= scalar;
				return *this;
			}
			__host__ __device__ constexpr vec2& operator/=(const vec2 & scalar)
			{
				x /= scalar.x;
				y /= scalar.y;
				return *this;
			}
			__host__ __device__ constexpr vec2& operator=(const vec2 & v)
			{
				x = v.x;
				y = v.y;
				return *this;
			}
			__host__ constexpr vec2& operator=(const Math::vec2<T>&v)
			{
				x = v.x;
				y = v.y;
				return *this;
			}
			__host__ constexpr bool operator==(const Math::vec2<T>& v) const
			{
				return x == v.x && y == v.y;
			}
			__host__ constexpr bool operator!=(const Math::vec2<T>& v) const
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

		template <typename T = unsigned char>
		class Color {};
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
			__host__ __device__ constexpr Color(const float& value)
				: red(value)
				, green(value)
				, blue(value)
				, alpha(value)
			{}
			__host__ __device__ constexpr Color(
				const float& red,
				const float& green,
				const float& blue,
				const float& alpha)
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
			__device__ constexpr Color<float> operator*(const float& factor) const
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
			__device__ constexpr Color<float>& operator*=(const float& factor)
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
			__device__ static constexpr Color<float> Mix(
				const Color<float>& color1,
				const Color<float>& color2,
				const float& balance)
			{
				return Color<float>(
					color1.red * balance + color2.red * (1.0f - balance),
					color1.green * balance + color2.green * (1.0f - balance),
					color1.blue * balance + color2.blue * (1.0f - balance),
					color1.alpha * balance + color2.alpha * (1.0f - balance));
			}


		public:
			__device__ constexpr void Mix(const Color<float>& color)
			{
				this->red = (this->red + color.red) / 2.0f;
				this->green = (this->green + color.green) / 2.0f;
				this->blue = (this->blue + color.blue) / 2.0f;
				this->alpha = (this->alpha + color.alpha) / 2.0f;
			}
			__device__ constexpr void Mix(const Color<float>& color, const float& balance)
			{
				this->red = (this->red * balance + color.red * (1.0f - balance));
				this->green = (this->green * balance + color.green * (1.0f - balance));
				this->blue = (this->blue * balance + color.blue * (1.0f - balance));
				this->alpha = (this->alpha * balance + color.alpha * (1.0f - balance));
			}


		public:
			__host__ __device__ constexpr void Set(
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
				const unsigned char& red,
				const unsigned char& green,
				const unsigned char& blue,
				const unsigned char& alpha = 0xFF)
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
			__device__ constexpr Color<unsigned char>& operator*=(const float& factor)
			{
				this->red = static_cast<unsigned char>(this->red * factor);
				this->green = static_cast<unsigned char>(this->green * factor);
				this->blue = static_cast<unsigned char>(this->blue * factor);
				this->alpha = static_cast<unsigned char>(this->alpha * factor);
				return *this;
			}
			__device__ constexpr Color<unsigned char> operator*(const float& factor) const
			{
				return Color<unsigned char>(
					static_cast<unsigned char>(this->red * factor),
					static_cast<unsigned char>(this->green * factor),
					static_cast<unsigned char>(this->blue * factor),
					static_cast<unsigned char>(this->alpha * factor));
			}


		public:
			__device__ static Color<unsigned char> Mix(const Color<unsigned char>& color1, const Color<unsigned char>& color2)
			{
				return Color<unsigned char>(
					(color1.red + color2.red) / 2,
					(color1.green + color2.green) / 2,
					(color1.blue + color2.blue) / 2,
					(color1.alpha + color2.alpha) / 2);
			}
			__device__ static Color<unsigned char> Mix(const Color<unsigned char>& color1, const Color<unsigned char>& color2, const unsigned char balance)
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
			__device__ void Mix(const Color<unsigned char>& color)
			{
				this->red = (this->red + color.red) / 2;
				this->green = (this->green + color.green) / 2;
				this->blue = (this->blue + color.blue) / 2;
				this->alpha = (this->alpha + color.alpha) / 2;
			}
			__device__ void Mix(const Color<unsigned char>& color, const unsigned char balance)
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
		typedef Color<unsigned char> ColorU;

		class CudaWorld;
		struct CudaMaterial;
		typedef vec2f CudaTexcrd;

		struct SeedThread
		{
			uint8_t seed;

			__device__ __inline__ SeedThread()
				: seed(0u)
			{}
			__device__ __inline__ void SetSeed(const uint8_t& s)
			{
				seed = s;
			}
		};
		struct BlockThread
		{
			uint32_t in_block_idx;
			uint32_t block_idx;

			__device__ __inline__ BlockThread()
				: in_block_idx(threadIdx.y* blockDim.x + threadIdx.x)
				, block_idx(blockIdx.y* gridDim.x + blockIdx.x)
			{}
		};
		struct GridThread
		{
			vec2ui32 in_grid;
			uint32_t in_grid_idx;

			__device__ __inline__ GridThread()
				: in_grid(blockIdx.x* blockDim.x + threadIdx.x, blockIdx.y* blockDim.y + threadIdx.y)
				, in_grid_idx(
					(blockIdx.y* gridDim.x + blockIdx.x)*
					(blockDim.x* blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x)
			{}
		};
		struct FullThread
			: public SeedThread
			, public BlockThread
			, public GridThread
		{};

		struct RNG
		{
		public:
			static constexpr uint32_t s_count = 0x400;
		private:
			float m_unsigned_uniform[s_count];

		public:
			__host__ void Reconstruct();

			__device__ __inline__ float GetUnsignedUniform(FullThread& thread) const
			{
				thread.seed += 1u;
				return m_unsigned_uniform[
					(thread.in_grid_idx + thread.seed) % RNG::s_count];
			}
		};
		struct Seeds
		{
		public:
			static constexpr uint32_t s_count = 0x100;
		public:
			uint8_t m_seeds[s_count];


			__host__ void Reconstruct(cudaStream_t& stream);

			__device__ __inline__ uint8_t GetSeed(const uint8_t& id) const
			{
				return m_seeds[id];
			}
			__device__ __inline__ void SetSeed(const uint8_t& id, const uint8_t& value)
			{
				m_seeds[id] = value;
			}
		};

		struct CudaConstantKernel
		{
		private:
			RNG m_rng;


		public:
			__host__ void Reconstruct();

			__device__ __inline__ const RNG& GetRNG() const
			{
				return m_rng;
			}
		};
		class CudaGlobalKernel
		{
		private:
			uint32_t m_render_idx;
			Seeds m_seeds;


		public:
			__host__ CudaGlobalKernel();
			__host__ CudaGlobalKernel(const CudaGlobalKernel&) = delete;
			__host__ CudaGlobalKernel(CudaGlobalKernel&&) = delete;
			__host__ ~CudaGlobalKernel();

		public:
			__host__ void Reconstruct(
				uint32_t render_idx,
				cudaStream_t& stream);

			__device__ __inline__ const uint32_t& GetRenderIdx() const
			{
				return m_render_idx;
			}
			__device__ __inline__ Seeds& GetSeeds()
			{
				return m_seeds;
			}
		};


		struct PathNode
		{
		public:
			vec3f point;


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
			static constexpr unsigned int MaxPathDepth = 8u;
			//PathNode pathNodes[MaxPathDepth];
			int currentNodeIndex;
			Color<float> finalColor;

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
				finalColor = Color<float>(0.0f, 0.0f, 0.0f, 1.0f);
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
			__device__ __inline__ void EndPath()
			{
				currentNodeIndex = MaxPathDepth - 1u;
			}
			__device__ __inline__ Color<float> CalculateFinalColor()
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
			vec3f origin;
			vec3f direction;
			float length;


		public:
			__device__ CudaRay()
				: length(3.402823466e+30f)
			{}
			__device__ CudaRay(
				const vec3f& origin,
				const vec3f& direction,
				const float& length = 3.402823466e+30f)
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
			const CudaMaterial* material;


		public:
			__device__ CudaSceneRay()
				: CudaRay()
				, material(nullptr)
			{}
			__device__ CudaSceneRay(
				const vec3f& origin,
				const vec3f& direction,
				const CudaMaterial* material,
				const float& length = 3.402823466e+30f)
				: CudaRay(origin, direction, length)
				, material(material)
			{}
			__device__ ~CudaSceneRay()
			{}
		};


		struct RayIntersection
		{
			CudaSceneRay ray;
			vec3f point;
			vec3f surface_normal;
			vec3f mapped_normal;
			Color<float> surface_color;
			float surface_emittance = 0.0f;
			float surface_reflectance = 0.0f;

			const CudaMaterial* surface_material;
			const CudaMaterial* behind_material;
			CudaTexcrd texcrd;

			float bvh_factor = 1.0f;


			__device__ RayIntersection()
				: surface_material(nullptr)
				, behind_material(nullptr)
			{}
		};
		struct CudaTriangle;
		struct TriangleIntersection
		{
			CudaRay ray;
			const CudaTriangle* triangle;
			float b1, b2;

			float bvh_factor = 1.0f;

			__device__ TriangleIntersection()
				: triangle(nullptr)
				, b1(0.0f), b2(0.0f)
			{}
		};


		struct __align__(16u) CudaTriangle
		{
		public:
			vec3f* v1, * v2, * v3;
			CudaTexcrd* t1, * t2, * t3;
			vec3f* n1, * n2, * n3;
			vec3f normal;
			uint32_t material_id;


		public:
			__host__ CudaTriangle(const Triangle & hostTriangle);


		public:
			__device__ __inline__ bool ClosestIntersection(TriangleIntersection & intersection) const
			{
				const vec3f edge1 = *v2 - *v1;
				const vec3f edge2 = *v3 - *v1;

				const vec3f pvec = vec3f::CrossProduct(intersection.ray.direction, edge2);

				float det = (vec3f::DotProduct(edge1, pvec));
				det += static_cast<float>(det > -1.0e-7f && det < 1.0e-7f) * 1.0e-7f;
				const float inv_det = 1.0f / det;

				const vec3f tvec = intersection.ray.origin - *v1;
				const float b1 = vec3f::DotProduct(tvec, pvec) * inv_det;
				if (b1 < 0.0f || b1 > 1.0f)
					return false;

				const vec3f qvec = vec3f::CrossProduct(tvec, edge1);

				const float b2 = vec3f::DotProduct(intersection.ray.direction, qvec) * inv_det;
				if (b2 < 0.0f || b1 + b2 > 1.0f)
					return false;

				const float t = vec3f::DotProduct(edge2, qvec) * inv_det;
				if (t <= 0.0f || t >= intersection.ray.length)
					return false;

				intersection.ray.length = t;
				intersection.triangle = this;
				intersection.b1 = b1;
				intersection.b2 = b2;

				return true;
			}
			__device__ __inline__ CudaTexcrd TexcrdFromBarycenter(
				const float& b1, const float& b2) const
			{
				if (!t1 || !t2 || !t3) return CudaTexcrd(0.5f, 0.5f);

				const float b3 = 1.0f - b1 - b2;
				const float u = t1->x * b3 + t2->x * b1 + t3->x * b2;
				const float v = t1->y * b3 + t2->y * b1 + t3->y * b2;
				return CudaTexcrd(u, v);
			}
			__device__ __inline__ void AverageNormal(
				const TriangleIntersection& intersection,
				vec3f& averaged_normal) const
			{
				if (!n1 || !n2 || !n3)
				{
					averaged_normal = normal;
					return;
				}

				averaged_normal = 
					(*n1 * (1.0f - intersection.b1 - intersection.b2) +
					*n2 * intersection.b1 +
					*n3 * intersection.b2).Normalized();
			}
			__device__ __inline__ void MapNormal(
				const ColorF& map_color,
				vec3f& mapped_normal) const
			{
				if (!t1 || !t2 || !t3) return;

				const vec3f edge1 = *v2 - *v1;
				const vec3f edge2 = *v3 - *v1;
				const vec2f dUV1 = *t2 - *t1;
				const vec2f dUV2 = *t3 - *t1;

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


		struct CudaCoordSystem
		{
		public:
			vec3f x_axis, y_axis, z_axis;

		public:
			__host__ CudaCoordSystem(
				const Math::vec3f& x = Math::vec3f(1.0f, 0.0f, 0.0f),
				const Math::vec3f& y = Math::vec3f(0.0f, 1.0f, 0.0f),
				const Math::vec3f& z = Math::vec3f(0.0f, 0.0f, 1.0f))
				: x_axis(x)
				, y_axis(y)
				, z_axis(z)
			{}

		public:
			__host__ CudaCoordSystem& operator=(const CoordSystem& coordSystem)
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
		struct CudaTransformation
		{
		public:
			vec3f position, rotation, center, scale;
			CudaCoordSystem coord_system;

		public:
			__host__ CudaTransformation& operator=(const Transformation& t)
			{
				position = t.GetPosition();
				rotation = t.GetRotation();
				center = t.GetCenter();
				scale = t.GetScale();
				coord_system = t.GetCoordSystem();
				return *this;
			}

		public:
			__device__ __inline__ void TransformRayG2L(CudaRay& ray) const
			{
				ray.origin -= position;
				coord_system.TransformForward(ray.origin);
				ray.origin /= scale;
				ray.origin -= center;

				coord_system.TransformForward(ray.direction);
				ray.direction /= scale;
			}
			__device__ __inline__ void TransformVectorL2G(vec3f& v) const
			{
				v /= scale;
				coord_system.TransformBackward(v);
			}
		};

		struct CudaBoundingBox
		{
			vec3f min, max;

			__host__ CudaBoundingBox()
				: min(0.0f, 0.0f, 0.0f)
				, max(0.0f, 0.0f, 0.0f)
			{}
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

				float tmin = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
				float tmax = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));

				return !(tmax < 0.0f || tmin > tmax || tmin > ray.length);
			}
		};


		// ~~~~~~~~ Helper Functions ~~~~~~~~
		__device__ __inline__ vec3f ReflectVector(
			const vec3f& vI,
			const vec3f& vN)
		{
			return (vN * -2.0f * vec3f::DotProduct(vN, vI) + vI);
		}
		__device__ __inline__ float RayToPointDistance(
			const CudaRay& ray,
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
			const CudaRay& ray,
			const vec3f& P,
			vec3f& vOP,
			float& dOP,
			float& vOP_dot_vD,
			float& dPQ)
		{
			// O - ray origin
			// P - specified point
			// vD - ray direction
			// Q - closest point to P lying on ray

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
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	}
}

#endif // !CUDA_RENDER_PARTS_CUH