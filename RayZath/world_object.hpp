#ifndef WORLD_OBJECT_H
#define WORLD_OBJECT_H

#include "updatable.hpp"
#include "roho.hpp"

#include <string>

namespace RayZath::Engine
{
	template <typename T> struct ConStruct;

	class WorldObject;
	template<> struct ConStruct<WorldObject>;

	class WorldObject : public Updatable
	{
	protected:
		std::string m_name;


	protected:
		WorldObject(const WorldObject& other) = default;
		WorldObject(WorldObject&& other) = default;
		WorldObject(
			Updatable* updatable,
			const ConStruct<WorldObject>& con_struct);


	public:
		void name(const std::string& name);
		const std::string& name() const noexcept;
	};


	template<> struct ConStruct<WorldObject>
	{
	public:
		std::string name;


	public:
		ConStruct(const ConStruct& conStruct)
			: name(conStruct.name)
		{}
		ConStruct(const std::string& name = "name")
			: name(name)
		{}
	};
}

#endif // !OBJECT_H
