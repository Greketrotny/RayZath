#include "world_object.h"

namespace RayZath
{
	WorldObject::WorldObject(const ConStruct<WorldObject>& con_struct, Updatable* updatable)
		: Updatable(updatable)
		, m_name(con_struct.name)
	{}
	WorldObject::~WorldObject()
	{}

	void WorldObject::SetName(const std::wstring& newName)
	{
		m_name = newName;
	}
	const std::wstring& WorldObject::GetName() const
	{
		return m_name;
	}
}