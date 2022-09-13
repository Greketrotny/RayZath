#include "group.hpp"

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

	void Group::link(Handle<Group> group, Handle<Mesh> object)
	{
		if (!group || !object) return;
		if (object->group()) object->group()->removeObject(object);
		object->setGroup(group);
		group->addObject(object);
	}
	void Group::unlink(Handle<Group> group, Handle<Mesh> object)
	{
		if (!group || !object) return;
		group->removeObject(object);
		object->setGroup({});
	}
	void Group::link(Handle<Group> group, Handle<Group> subgroup)
	{
		if (!group || !subgroup) return; // invalid handlers
		if (subgroup->group() == group) return; // already is in the group
		if (group->group() == subgroup) // currently subgroup is supergroup of group (circularity)
		{
			unlink(subgroup, group);
			auto supergroup = subgroup->group();
			unlink(supergroup, subgroup);
			link(supergroup, group);
			link(group, subgroup);
			return;
		}
		if (subgroup->group()) subgroup->group()->removeGroup(subgroup); // unlink from current parent
		subgroup->setGroup(group); // link to new parent
		group->addGroup(subgroup); // add subgroup to new parent
	}
	void Group::unlink(Handle<Group> group, Handle<Group> subgroup)
	{
		if (!group || !subgroup) return;
		group->removeGroup(subgroup);
		subgroup->setGroup({});
	}

	void Group::RequestUpdate()
	{
		for (auto& object : m_objects)
			if (object) object->stateRegister().RequestUpdate();
		for (auto& group : m_groups)
			if (group) group->RequestUpdate();
		stateRegister().MakeModified();
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
	void Group::addObject(const Handle<Mesh>& new_object)
	{
		if (!new_object) return;
		m_objects.push_back(new_object);

		for (auto& object : m_objects)
			if (object) object->stateRegister().RequestUpdate();
	}
	void Group::removeObject(const Handle<Mesh>& object)
	{
		m_objects.erase(std::remove(m_objects.begin(), m_objects.end(), object), m_objects.end());
	}
}