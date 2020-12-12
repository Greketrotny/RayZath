#include "render_object.h"
#include "rzexception.h"

namespace RayZath
{
	RenderObject::RenderObject(
		const size_t& id,
		Updatable* updatable, 
		const ConStruct<RenderObject>& conStruct)
		: WorldObject(id, updatable, conStruct)
	{
		SetPosition(conStruct.position);
		SetRotation(conStruct.rotation);
		SetCenter(conStruct.center);
		SetScale(conStruct.scale);
		SetMaterial(conStruct.material);
	}
	RenderObject::~RenderObject()
	{}

	void RenderObject::SetPosition(const Math::vec3<float>& newPosition)
	{
		m_position = newPosition;
		GetStateRegister().RequestUpdate();
	}
	void RenderObject::SetRotation(const Math::vec3<float>& newRotation)
	{
		m_rotation = newRotation;
		GetStateRegister().RequestUpdate();
	}
	void RenderObject::SetCenter(const Math::vec3<float>& newMeshCenter)
	{
		m_center = newMeshCenter;
		GetStateRegister().RequestUpdate();
	}
	void RenderObject::SetScale(const Math::vec3<float>& newScale)
	{
		m_scale = newScale;
		GetStateRegister().RequestUpdate();
	}
	void RenderObject::SetMaterial(Material* newMaterial)
	{
		if (newMaterial == nullptr)
			ThrowException(L"RenderObject::SetMaterial(): newMaterial was nullptr");

		mp_material = newMaterial;
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
	/*const Material* RenderObject::GetMaterial() const
	{
		return mp_material;
	}
	Material* RenderObject::GetMaterial()
	{
		return mp_material;
	}*/
	const Material& RenderObject::GetMaterial() const
	{
		return *mp_material;
	}
	Material& RenderObject::GetMaterial()
	{
		return *mp_material;
	}
	const BoundingBox& RenderObject::GetBoundingBox() const
	{
		return m_bounding_box;
	}
}