#ifndef RENDER_PARTS_H
#define RENDER_PARTS_H

#include "vec3.h"
#include "bitmap.h"

namespace RayZath
{
	struct Material
	{
	private:
		float m_reflectance;
		float m_glossiness;

		float m_transmitance;
		float m_ior;

		float m_emitance;


	public:
		Material(
			const float& reflectance = 0.0f,
			const float& glossiness = 0.0f,
			const float& transmitance = 0.0f,
			const float& m_ior = 1.0f,
			const float& emitance = 0.0f);
		Material(const Material& material);
		~Material();


	public:
		Material& operator=(const Material& material);


	public:
		void SetReflectance(const float& reflectance);
		void SetGlossiness(const float& glossiness);
		void SetTransmitance(const float& transmitance);
		void SetIndexOfRefraction(const float& ior);
		void SetEmitance(const float& emitance);

		float GetReflectance() const noexcept;
		float GetGlossiness() const noexcept;
		float GetTransmitance() const noexcept;
		float GetIndexOfRefraction() const noexcept;
		float GetEmitance() const noexcept;
	};

	struct Texcrd
	{
		float u, v;

		Texcrd(float u, float v)
			: u(u)
			, v(v)
		{}
	};
	struct Texture
	{
		enum class FilterMode
		{
			Point,
			Linear
		};
	private:
		Graphics::Bitmap m_bitmap;
		FilterMode m_filter_mode = FilterMode::Point;


	public:
		Texture() = delete;
		Texture(const Texture& texture);
		Texture(Texture&& texture) noexcept;
		Texture(size_t width, size_t height, FilterMode filterMode = FilterMode::Point);
		Texture(const Graphics::Bitmap& bitmap, FilterMode filterMode = FilterMode::Point);
		~Texture();


	public:
		Texture& operator=(const Texture& texture);
		Texture& operator=(Texture&& texture) noexcept;

		
	public:
		const Graphics::Bitmap& GetBitmap() const noexcept;
		FilterMode GetFilterMode() const noexcept;
	};

	struct Triangle
	{
		typedef Math::vec3<float> Vertex;
	private:
		Vertex* v1 = nullptr, * v2 = nullptr, * v3 = nullptr;
		Vertex normal;
		Texcrd* t1 = nullptr, * t2 = nullptr, * t3 = nullptr;
		Graphics::Color color;


	private:
		Triangle(
			Vertex* v1, Vertex* v2, Vertex* v3,
			Texcrd* t1, Texcrd* t2, Texcrd* t3,
			Graphics::Color color = Graphics::Color(0xFF, 0xFF, 0xFF));
		~Triangle();


	public:
		// overall
		const Graphics::Color& Color();
		void Color(const Graphics::Color& newColor);
		const Math::vec3<float>& GetNormal() const;

		// vertices
		Vertex& V1() const;
		void V1(const Vertex& newVertex);
		void V1(const float& x, const float& y, const float& z);
		Vertex& V2() const;
		void V2(const Vertex& newVertex);
		void V2(const float& x, const float& y, const float& z);
		Vertex& V3() const;
		void V3(const Vertex& newVertex);
		void V3(const float& x, const float& y, const float& z);

		// texture coordinates
		Texcrd& T1() const;
		void T1(const Texcrd& newTexcds);
		void T1(const float& u, const float& v);
		Texcrd& T2() const;
		void T2(const Texcrd& newTexcds);
		void T2(const float& u, const float& v);
		Texcrd& T3() const;
		void T3(const Texcrd& newTexcds);
		void T3(const float& u, const float& v);

		// -- friends -- //
		friend class Mesh;
		friend class CudaMesh;
		friend struct CudaTriangleStorage;
		friend struct CudaTriangle;
	};
}

#endif // !RENDER_PARTS_H