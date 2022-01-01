#ifndef RENDER_OBJECT_H
#define RENDER_OBJECT_H

#include "world_object.h"
#include "render_parts.h"
#include "material.h"
#include "roho.h"

#include "vec3.h"

namespace RayZath::Engine
{
	class RenderObject;
	template<> struct ConStruct<RenderObject>;

	class RenderObject : public WorldObject
	{
	protected:
		Transformation m_transformation;
		BoundingBox m_bounding_box;


	public:
		RenderObject(
			Updatable* updatable, 
			const ConStruct<RenderObject>& conStruct);


		void SetPosition(const Math::vec3f& position);
		void SetRotation(const Math::vec3f& rotation);
		void SetCenter(const Math::vec3f& center);
		void SetScale(const Math::vec3f& scale);
		void LookAtPoint(const Math::vec3f& point, const Math::angle_radf& angle = 0.0f);
		void LookInDirection(const Math::vec3f& direction, const Math::angle_radf& angle = 0.0f);

		const Transformation& GetTransformation() const;
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
			const std::string& name = "name",
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