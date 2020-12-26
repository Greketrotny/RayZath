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
		Math::vec3<float> m_position;
		Math::vec3<float> m_rotation;
		Math::vec3<float> m_center;
		Math::vec3<float> m_scale;
		BoundingBox m_bounding_box;


	public:
		RenderObject(
			Updatable* updatable, 
			const ConStruct<RenderObject>& conStruct);
		~RenderObject();


		void SetPosition(const Math::vec3<float>& position);
		void SetRotation(const Math::vec3<float>& rotation);
		void SetCenter(const Math::vec3<float>& center);
		void SetScale(const Math::vec3<float>& scale);

		const Math::vec3<float>& GetPosition() const;
		const Math::vec3<float>& GetRotation() const;
		const Math::vec3<float>& GetCenter() const;
		const Math::vec3<float>& GetScale() const;
		const BoundingBox& GetBoundingBox() const;
	};


	template<> struct ConStruct<RenderObject> : public ConStruct<WorldObject>
	{
	public:
		Math::vec3<float> position;
		Math::vec3<float> rotation;
		Math::vec3<float> center;
		Math::vec3<float> scale;

	public:
		ConStruct(
			const std::wstring& name = L"name",
			const Math::vec3<float>& position = Math::vec3<float>(0.0f, 0.0f, 0.0f),
			const Math::vec3<float>& rotation = Math::vec3<float>(0.0f, 0.0f, 0.0f),
			const Math::vec3<float>& center = Math::vec3<float>(0.0f, 0.0f, 0.0f),
			const Math::vec3<float>& scale = Math::vec3<float>(1.0f, 1.0f, 1.0f))
			: ConStruct<WorldObject>(name)
			, position(position)
			, rotation(rotation)
			, center(center)
			, scale(scale)
		{}
	};
}

#endif