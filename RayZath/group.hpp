#ifndef GROUP_H
#define GROUP_H

#include "groupable.hpp"
#include "instance.hpp"

namespace RayZath::Engine
{
	class Group;
	template<> struct ConStruct<Group>;

	class Group : public WorldObject, public Groupable
	{
	private:
		Transformation m_transformation;
		BoundingBox m_bounding_box;

		std::vector<Handle<Group>> m_groups;
		std::vector<Handle<Instance>> m_objects;


	public:
		Group(const Group&) = delete;
		Group(Group&&) = delete;
		Group(
			Updatable* updatable,
			ConStruct<Group> construct);


	public:
		Group& operator=(const Group&) = delete;
		Group& operator=(Group&&) = delete;


	public:
		Transformation& transformation();
		const Transformation& transformation() const;
		const std::vector<Handle<Group>>& groups() const;
		const std::vector<Handle<Instance>>& objects() const;
		bool topLevelGroup();

		static void link(Handle<Group> group, Handle<Instance> object);
		static void unlink(Handle<Group> group, Handle<Instance> object);
		static void link(Handle<Group> group, Handle<Group> subgroup);
		static void unlink(Handle<Group> group, Handle<Group> subgroup);

		void RequestUpdate();

	private:
		void addGroup(const Handle<Group>& group);
		void removeGroup(const Handle<Group>& group);
		void addObject(const Handle<Instance>& object);
		void removeObject(const Handle<Instance>& object);
	};


	template<> struct ConStruct<Group> : public ConStruct<WorldObject>
	{
		Math::vec3f position;
		Math::vec3f rotation;
		Math::vec3f scale;

		ConStruct(
			std::string name = "name",
			const Math::vec3f& position = Math::vec3f(0.0f, 0.0f, 0.0f),
			const Math::vec3f& rotation = Math::vec3f(0.0f, 0.0f, 0.0f),
			const Math::vec3f& scale = Math::vec3f(1.0f, 1.0f, 1.0f))
			: ConStruct<WorldObject>(std::move(name))
			, position(position)
			, rotation(rotation)
			, scale(scale)
		{}
	};
}

#endif // !GROUP_H