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
		const uint32_t m_id;
		std::wstring m_name;


	protected:
		WorldObject(
			const uint32_t& id,
			Updatable* updatable,
			const ConStruct<WorldObject>& con_struct);
	public:
		virtual ~WorldObject();


	public:
		void SetName(const std::wstring& newName);
		const std::wstring& GetName() const noexcept;
		uint32_t GetId() const noexcept;


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