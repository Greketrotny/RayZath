#ifndef RENDER_PARTS_H
#define RENDER_PARTS_H

#include "vec3.h"
#include "bitmap.h"
#include "world_object.h"

namespace RayZath
{
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
	struct Texture : public WorldObject
	{
	public:
		enum class FilterMode
		{
			Point,
			Linear
		};
	private:
		Graphics::Bitmap m_bitmap;
		FilterMode m_filter_mode = FilterMode::Point;


	public:
		Texture(const Texture& texture) = delete;
		Texture(Texture&& texture) = delete;
		Texture(
			Updatable* updatable,
			const ConStruct<Texture>& con_struct);


	public:
		Texture& operator=(const Texture& texture) = delete;
		Texture& operator=(Texture&& texture) = delete;

		
	public:
		const Graphics::Bitmap& GetBitmap() const noexcept;
		FilterMode GetFilterMode() const noexcept;

		void SetBitmap(const Graphics::Bitmap& bitmap);
		void SetFilterMode(const FilterMode filter_mode);
	};
	template <> struct ConStruct<Texture> : public ConStruct<WorldObject>
	{
		Graphics::Bitmap bitmap;
		Texture::FilterMode filter_mode;

		ConStruct(
			const std::wstring& name = L"name",
			const Graphics::Bitmap& bitmap = Graphics::Bitmap(64u, 64u),
			const Texture::FilterMode& filter_mode = Texture::FilterMode::Point)
			: ConStruct<WorldObject>(name)
			, bitmap(bitmap)
			, filter_mode(filter_mode)
		{}
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