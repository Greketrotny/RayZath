#include "group.h"

namespace RayZath::Engine
{
	Group::Group(
		Updatable* updatable,
		ConStruct<Group> construct)
		: WorldObject(updatable, construct)
		, m_transformation(construct.position, construct.rotation, construct.scale)
	{}


	Transformation& Group::transformation()
	{
		return m_transformation;
	}
	const Transformation& Group::transformation() const
	{
		return m_transformation;
	}
	const std::vector<Handle<Group>>& Group::groups() const
	{
		return m_groups;
	}
	const std::vector<Handle<Mesh>>& Group::objects() const
	{
		return m_objects;
	}
	bool Group::topLevelGroup()
	{
		return !group();
	}

	void Group::link(const Handle<Group>& group, const Handle<Mesh>& object)
	{
		if (!group || !object) return;
		object->setGroup(group);
		group->addObject(object);
	}
	void Group::unlink(const Handle<Group>& group, const Handle<Mesh>& object)
	{
		if (!group || !object) return;
		object->setGroup({});
		group->removeObject(object);
	}
	void Group::link(const Handle<Group>& group, const Handle<Group>& subgroup)
	{
		if (!group || !subgroup) return;
		subgroup->setGroup(group);
		group->addGroup(subgroup);
	}
	void Group::unlink(const Handle<Group>& group, const Handle<Group>& subgroup)
	{
		if (!group || !subgroup) return;
		subgroup->setGroup({});
		group->removeGroup(subgroup);
	}

	void Group::RequestUpdate()
	{
		for (auto& object : m_objects)
			if (object) object->GetStateRegister().RequestUpdate();
		GetStateRegister().MakeModified();
	}

	void Group::addGroup(const Handle<Group>& group)
	{
		if (!group) return;
		m_groups.push_back(group);
	}
	void Group::removeGroup(const Handle<Group>& group)
	{
		m_groups.erase(std::remove(m_groups.begin(), m_groups.end(), group), m_groups.end());
	}
	void Group::addObject(const Handle<Mesh>& object)
	{
		if (!object) return;
		m_objects.push_back(object);

		for (auto& object : m_objects)
			if (object) object->GetStateRegister().RequestUpdate();
	}
	void Group::removeObject(const Handle<Mesh>& object)
	{
		m_objects.erase(std::remove(m_objects.begin(), m_objects.end(), object), m_objects.end());
	}
}