#ifndef WORLD_OBJECT_H
#define WORLD_OBJECT_H

#include "updatable.h"
#include "roho.h"

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
		WorldObject(
			Updatable* updatable,
			const ConStruct<WorldObject>& con_struct);


	public:
		void SetName(const std::string& name);
		const std::string& GetName() const noexcept;
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