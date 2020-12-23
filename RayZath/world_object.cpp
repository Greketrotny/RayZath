#include "world_object.h"

namespace RayZath
{
	WorldObject::WorldObject(
		Updatable* updatable,
		const ConStruct<WorldObject>& con_struct)
		: Updatable(updatable)
		, m_name(con_struct.name)
	{}

	void WorldObject::SetName(const std::wstring& name)
	{
		m_name = name;
	}
	const std::wstring& WorldObject::GetName() const noexcept
	{
		return m_name;
	}
}