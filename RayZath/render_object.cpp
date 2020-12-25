#include "render_object.h"
#include "rzexception.h"

namespace RayZath
{
	RenderObject::RenderObject(
		Updatable* updatable, 
		const ConStruct<RenderObject>& conStruct)
		: WorldObject(updatable, conStruct)
	{
		SetPosition(conStruct.position);
		SetRotation(conStruct.rotation);
		SetCenter(conStruct.center);
		SetScale(conStruct.scale);
	}
	RenderObject::~RenderObject()
	{}

	void RenderObject::SetPosition(const Math::vec3<float>& position)
	{
		m_position = position;
		GetStateRegister().RequestUpdate();
	}
	void RenderObject::SetRotation(const Math::vec3<float>& rotation)
	{
		m_rotation = rotation;
		GetStateRegister().RequestUpdate();
	}
	void RenderObject::SetCenter(const Math::vec3<float>& center)
	{
		m_center = center;
		GetStateRegister().RequestUpdate();
	}
	void RenderObject::SetScale(const Math::vec3<float>& scale)
	{
		m_scale = scale;
		GetStateRegister().RequestUpdate();
	}

	const Math::vec3<float>& RenderObject::GetPosition() const
	{
		return m_position;
	}
	const Math::vec3<float>& RenderObject::GetRotation() const
	{
		return m_rotation;
	}
	const Math::vec3<float>& RenderObject::GetCenter() const
	{
		return m_center;
	}
	const Math::vec3<float>& RenderObject::GetScale() const
	{
		return m_scale;
	}
	const BoundingBox& RenderObject::GetBoundingBox() const
	{
		return m_bounding_box;
	}
}