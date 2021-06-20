#include "world_object.h"

namespace RayZath
{
	WorldObject::WorldObject(
		Updatable* updatable,
		const ConStruct<WorldObject>& con_struct)
		: Updatable(updatable)
		, m_name(con_struct.name)
	{}

	void WorldObject::SetName(const std::string& name)
	{
		m_name = name;
	}
	const std::string& WorldObject::GetName() const noexcept
	{
		return m_name;
	}
}