#ifndef GROUPABLE_H
#define GROUPABLE_H

#include "render_parts.hpp"

#include <vector>

namespace RayZath::Engine
{
	class Group;
	class Groupable
	{
		Handle<Group> m_group;
	protected:
		Transformation m_transformation_in_group;

	public:
		Groupable(const Groupable&) = delete;
		Groupable(Groupable&&) = delete;
		Groupable(Handle<Group> group = {});

		Group& operator=(const Group&) = delete;
		Group& operator=(Group&&) = delete;

		void setGroup(const Handle<Group>& group);
		const Handle<Group>& group() const;
		const Transformation& transformationInGroup() const;
	};
}

#endif // !MESH_H