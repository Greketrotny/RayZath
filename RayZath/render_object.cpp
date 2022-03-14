#include "render_object.h"
#include "rzexception.h"

namespace RayZath::Engine
{
	RenderObject::RenderObject(
		Updatable* updatable,
		const ConStruct<RenderObject>& conStruct)
		: WorldObject(updatable, conStruct)
		, m_transformation(
			conStruct.position, 
			conStruct.rotation,
			conStruct.scale)
	{}


	void RenderObject::SetPosition(const Math::vec3f& position)
	{
		m_transformation.SetPosition(position);
		GetStateRegister().RequestUpdate();
	}
	void RenderObject::SetRotation(const Math::vec3f& rotation)
	{
		m_transformation.SetRotation(rotation);
		GetStateRegister().RequestUpdate();
	}
	void RenderObject::SetScale(const Math::vec3f& scale)
	{
		m_transformation.SetScale(scale);
		GetStateRegister().RequestUpdate();
	}
	void RenderObject::LookAtPoint(
		const Math::vec3f& point, 
		const Math::angle_radf& angle)
	{
		m_transformation.LookAtPoint(point, angle);
		GetStateRegister().RequestUpdate();
	}
	void RenderObject::LookInDirection(
		const Math::vec3f& direction,
		const Math::angle_radf& angle)
	{
		m_transformation.LookInDirection(direction, angle);
		GetStateRegister().RequestUpdate();
	}

	const Transformation& RenderObject::GetTransformation() const
	{
		return m_transformation;
	}
	const BoundingBox& RenderObject::GetBoundingBox() const
	{
		return m_bounding_box;
	}
}