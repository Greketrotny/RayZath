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
		enum class AddressMode
		{
			Wrap,
			Clamp,
			Mirror,
			Border
		};
	private:
		Graphics::Bitmap m_bitmap;
		FilterMode m_filter_mode;
		AddressMode m_address_mode;


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
		AddressMode GetAddressMode() const noexcept;

		void SetBitmap(const Graphics::Bitmap& bitmap);
		void SetFilterMode(const FilterMode filter_mode);
		void SetAddressMode(const AddressMode address_mode);
	};
	template <> struct ConStruct<Texture> : public ConStruct<WorldObject>
	{
		Graphics::Bitmap bitmap;
		Texture::FilterMode filter_mode;
		Texture::AddressMode address_mode;

		ConStruct(
			const std::wstring& name = L"name",
			const Graphics::Bitmap& bitmap = Graphics::Bitmap(64u, 64u),
			const Texture::FilterMode& filter_mode = Texture::FilterMode::Point,
			const Texture::AddressMode& address_mode = Texture::AddressMode::Wrap)
			: ConStruct<WorldObject>(name)
			, bitmap(bitmap)
			, filter_mode(filter_mode)
			, address_mode(address_mode)
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
		uint32_t material_id;


	public:
		Triangle(
			Vertex* v1, Vertex* v2, Vertex* v3,
			Texcrd* t1, Texcrd* t2, Texcrd* t3,
			Normal* n1, Normal* n2, Normal* n3,
			const uint32_t& mat_id);
		~Triangle();


	public:
		void CalculateNormal();
		BoundingBox GetBoundingBox() const;
	};
}

#endif // !RENDER_PARTS_H