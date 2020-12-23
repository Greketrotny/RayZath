#ifndef WORLD_OBJECT_H
#define WORLD_OBJECT_H

#include "updatable.h"
#include "roho.h"

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
		WorldObject(
			Updatable* updatable,
			const ConStruct<WorldObject>& con_struct);


	public:
		void SetName(const std::wstring& name);
		const std::wstring& GetName() const noexcept;
	};


	template<> struct ConStruct<WorldObject>
	{
	public:
		std::wstring name;


	public:
		ConStruct(const ConStruct& conStruct)
			: name(conStruct.name)
		{}
		ConStruct(const std::wstring& name = L"name")
			: name(name)
		{}
	};
}

#endif // !OBJECT_H