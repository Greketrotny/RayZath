#ifndef RENDER_OBJECT_H
#define RENDER_OBJECT_H

#include "world_object.h"
#include "render_parts.h"
#include "material.h"
#include "roho.h"

#include "vec3.h"

namespace RayZath
{
	class RenderObject;
	template<> struct ConStruct<RenderObject>;

	class RenderObject : public WorldObject
	{
	protected:
		Math::vec3f m_position;
		Math::vec3f m_rotation;
		Math::vec3f m_center;
		Math::vec3f m_scale;
		BoundingBox m_bounding_box;


	public:
		RenderObject(
			Updatable* updatable, 
			const ConStruct<RenderObject>& conStruct);
		~RenderObject();


		void SetPosition(const Math::vec3f& position);
		void SetRotation(const Math::vec3f& rotation);
		void SetCenter(const Math::vec3f& center);
		void SetScale(const Math::vec3f& scale);

		const Math::vec3f& GetPosition() const;
		const Math::vec3f& GetRotation() const;
		const Math::vec3f& GetCenter() const;
		const Math::vec3f& GetScale() const;
		const BoundingBox& GetBoundingBox() const;
	};


	template<> struct ConStruct<RenderObject> : public ConStruct<WorldObject>
	{
	public:
		Math::vec3f position;
		Math::vec3f rotation;
		Math::vec3f center;
		Math::vec3f scale;

	public:
		ConStruct(
			const std::wstring& name = L"name",
			const Math::vec3f& position = Math::vec3f(0.0f, 0.0f, 0.0f),
			const Math::vec3f& rotation = Math::vec3f(0.0f, 0.0f, 0.0f),
			const Math::vec3f& center = Math::vec3f(0.0f, 0.0f, 0.0f),
			const Math::vec3f& scale = Math::vec3f(1.0f, 1.0f, 1.0f))
			: ConStruct<WorldObject>(name)
			, position(position)
			, rotation(rotation)
			, center(center)
			, scale(scale)
		{}
	};
}

#endif