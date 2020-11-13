#include "world_object.h"

namespace RayZath
{
	WorldObject::WorldObject(
		const uint32_t& id,
		Updatable* updatable,
		const ConStruct<WorldObject>& con_struct)
		: m_id(id)
		, Updatable(updatable)
		, m_name(con_struct.name)
	{}
	WorldObject::~WorldObject()
	{}

	void WorldObject::SetName(const std::wstring& newName)
	{
		m_name = newName;
	}
	const std::wstring& WorldObject::GetName() const noexcept
	{
		return m_name;
	}
	uint32_t WorldObject::GetId() const noexcept
	{
		return m_id;
	}
}