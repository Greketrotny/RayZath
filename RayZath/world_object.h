#ifndef WORLD_OBJECT_H
#define WORLD_OBJECT_H

#include "updatable.h"

#include <string>

namespace RayZath
{
	template <typename T> struct ConStruct;

	class WorldObject;
	template<> struct ConStruct<WorldObject>;

	class WorldObject : public Updatable
	{
	protected:
		std::wstring m_name;


	protected:
		WorldObject(const ConStruct<WorldObject>& con_struct, Updatable* updatable);
	public:
		virtual ~WorldObject();


	public:
		void SetName(const std::wstring& newName);
		const std::wstring& GetName() const;


		friend class World;
	};


	template<> struct ConStruct<WorldObject>
	{
	public:
		std::wstring name;


	public:
		ConStruct(const ConStruct& conStruct)
			: name(conStruct.name)
		{}
		ConStruct(std::wstring name = L"WorldObjectName")
			: name(name)
		{}
		~ConStruct()
		{}
	};
}

#endif // !OBJECT_H