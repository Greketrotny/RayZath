#ifndef RENDER_OBJECT_H
#define RENDER_OBJECT_H

#include "world_object.h"
#include "render_parts.h"

#include "vec3.h"

namespace RayZath
{
	class RenderObject;
	template<> struct ConStruct<RenderObject>;

	class RenderObject : public WorldObject
	{
	protected:
		Math::vec3<float> m_position;
		Math::vec3<float> m_rotation;
		Math::vec3<float> m_center;
		Math::vec3<float> m_scale;
		Material m_material;
		BoundingBox m_bounding_box;


	public:
		RenderObject(
			const size_t& id,
			Updatable* updatable, 
			const ConStruct<RenderObject>& conStruct);
		~RenderObject();


	public:
		void SetPosition(const Math::vec3<float>& position);
		void SetRotation(const Math::vec3<float>& rotation);
		void SetCenter(const Math::vec3<float>& center);
		void SetScale(const Math::vec3<float>& scale);
		void SetMaterial(const Material& material);

		const Math::vec3<float>& GetPosition() const;
		const Math::vec3<float>& GetRotation() const;
		const Math::vec3<float>& GetCenter() const;
		const Math::vec3<float>& GetScale() const;
		const Material& GetMaterial() const;
		Material& GetMaterial();
		const BoundingBox& GetBoundingBox() const;

	public:
		friend struct CudaBoundingBox;
	};


	template<> struct ConStruct<RenderObject> : public ConStruct<WorldObject>
	{
	public:
		Math::vec3<float> position;
		Math::vec3<float> rotation;
		Math::vec3<float> center;
		Math::vec3<float> scale;
		Material material;

	public:
		ConStruct(
			const ConStruct<WorldObject>& conStruct = ConStruct<WorldObject>(),
			const Math::vec3<float>& position = Math::vec3<float>(0.0f, 0.0f, 0.0f),
			const Math::vec3<float>& rotation = Math::vec3<float>(0.0f, 0.0f, 0.0f),
			const Math::vec3<float>& center = Math::vec3<float>(0.0f, 0.0f, 0.0f),
			const Math::vec3<float>& scale = Math::vec3<float>(1.0f, 1.0f, 1.0f),
			const Material& material = Material())
			: ConStruct<WorldObject>(conStruct)
			, position(position)
			, rotation(rotation)
			, center(center)
			, scale(scale)
			, material(material)
		{}
		~ConStruct()
		{}
	};
}

#endif