#include "world_object.hpp"

namespace RayZath::Engine
{
	WorldObject::WorldObject(
		Updatable* updatable,
		const ConStruct<WorldObject>& con_struct)
		: Updatable(updatable)
		, m_name(con_struct.name)
	{}

	void WorldObject::name(const std::string& name)
	{
		m_name = name;
	}
	const std::string& WorldObject::name() const noexcept
	{
		return m_name;
	}
}