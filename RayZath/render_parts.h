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

		float m_transmittance;
		float m_ior;

		float m_emittance;
		float m_scattering;


	public:
		Material(
			const float& reflectance = 0.0f,
			const float& glossiness = 0.0f,
			const float& transmittance = 0.0f,
			const float& m_ior = 1.0f,
			const float& emittance = 0.0f,
			const float& scattering = 0.0f);
		Material(const Material& material);
		~Material();


	public:
		Material& operator=(const Material& material);


	public:
		void SetReflectance(const float& reflectance);
		void SetGlossiness(const float& glossiness);
		void SetTransmittance(const float& transmittance);
		void SetIndexOfRefraction(const float& ior);
		void SetEmittance(const float& emittance);
		void SetScattering(const float& scattering);

		float GetReflectance() const noexcept;
		float GetGlossiness() const noexcept;
		float GetTransmittance() const noexcept;
		float GetIndexOfRefraction() const noexcept;
		float GetEmittance() const noexcept;
		float GetScattering() const noexcept;
	};

	struct BoundingBox
	{
	public:
		Math::vec3<float> min, max;


	public:
		BoundingBox(
			const Math::vec3<float>& p1 = Math::vec3<float>(0.0f, 0.0f, 0.0f),
			const Math::vec3<float>& p2 = Math::vec3<float>(0.0f, 0.0f, 0.0f));
		BoundingBox(
			const Math::vec3<float>& p1,
			const Math::vec3<float>& p2,
			const Math::vec3<float>& p3);


		void Reset(const Math::vec3<float>& point = Math::vec3<float>(0.0f, 0.0f, 0.0f));
		void ExtendBy(const Math::vec3<float>& point);
		void ExtendBy(const BoundingBox& bb);

		Math::vec3<float> GetCentroid() const noexcept;
	};

	struct Texcrd
	{
		float u, v;

		Texcrd(float u = 0.0f, float v = 0.0f)
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

	typedef Math::vec3<float> Vertex;
	typedef Math::vec3<float> Normal;
	struct Triangle
	{
	public:
		Vertex *v1, *v2, *v3;
		Texcrd *t1, *t2, *t3;
		Normal *n1, *n2, *n3;
		Math::vec3<float> normal;
		Graphics::Color color;


	public:
		Triangle(
			Vertex* v1, Vertex* v2, Vertex* v3,
			Texcrd* t1, Texcrd* t2, Texcrd* t3,
			Normal* n1, Normal* n2, Normal* n3,
			Graphics::Color color = Graphics::Color(0xFF, 0xFF, 0xFF));
		~Triangle();


	public:
		void CalculateNormal();
		BoundingBox GetBoundingBox() const;
	};
}

#endif // !RENDER_PARTS_H