#include "groupable.hpp"

#include "group.hpp"

namespace RayZath::Engine
{
	Groupable::Groupable(Handle<Group> group)
		: m_group(std::move(group))
	{}

	void Groupable::setGroup(const Handle<Group>& group)
	{
		m_group = group;
	}
	const Handle<Group>& Groupable::group() const
	{
		return m_group;
	}
	const Transformation& Groupable::transformationInGroup() const
	{
		return m_transformation_in_group;
	}
}