#include "plane.h"

namespace RayZath
{
	// ~~~~~~~~ [CLASS] Plane ~~~~~~~~
	Plane::Plane(
		Updatable* updatable,
		const ConStruct<Plane>& con_struct)
		: RenderObject(updatable, con_struct)
		, m_material(con_struct.material, std::bind(&Plane::NotifyMaterial, this))
	{}


	void Plane::SetMaterial(const Handle<Material>& material)
	{
		m_material = material;
		GetStateRegister().MakeModified();
	}
	const Handle<Material>& Plane::GetMaterial() const noexcept
	{
		return m_material;
	}

	void Plane::NotifyMaterial()
	{
		GetStateRegister().MakeModified();
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}