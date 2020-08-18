#include "render_object.h"

namespace RayZath
{
	RenderObject::RenderObject(
		const size_t& id,
		Updatable* updatable, 
		const ConStruct<RenderObject>& conStruct)
		: WorldObject(id, updatable, conStruct)
	{
		SetPosition(conStruct.position);
		SetRotation(m_rotation = conStruct.rotation);
		SetCenter(m_center = conStruct.center);
		SetScale(m_scale = conStruct.scale);
		SetMaterial(m_material = conStruct.material);
	}
	RenderObject::~RenderObject()
	{}

	void RenderObject::SetPosition(const Math::vec3<float>& newPosition)
	{
		m_position = newPosition;
		RequestUpdate();
	}
	void RenderObject::SetRotation(const Math::vec3<float>& newRotation)
	{
		m_rotation = newRotation;
		RequestUpdate();
	}
	void RenderObject::SetCenter(const Math::vec3<float>& newMeshCenter)
	{
		m_center = newMeshCenter;
		RequestUpdate();
	}
	void RenderObject::SetScale(const Math::vec3<float>& newScale)
	{
		m_scale = newScale;
		RequestUpdate();
	}
	void RenderObject::SetMaterial(const Material& newMaterial)
	{
		m_material = newMaterial;
		RequestUpdate();
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
	const Material& RenderObject::GetMaterial() const
	{
		return m_material;
	}
}