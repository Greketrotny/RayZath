#include "updatable.h"
#include <memory>

namespace RayZath
{
	// ~~~~~~~~ [STRUCT] StateRegister ~~~~~~~~
	StateRegister::StateRegister(StateRegister&& other)
		: mp_parent(other.mp_parent)
		, m_modified(other.m_modified)
		, m_requires_update(other.m_requires_update)
	{
		other.mp_parent = nullptr;
		m_modified = false;
		m_requires_update = false;
	}
	StateRegister::StateRegister(Updatable* parent)
		: mp_parent(parent)
		, m_modified(true)
		, m_requires_update(true)
	{}

	void StateRegister::MakeModified()
	{
		m_modified = true;
		if (mp_parent) mp_parent->GetStateRegister().MakeModified();
	}
	void StateRegister::RequestUpdate()
	{
		m_requires_update = true;
		m_modified = true;
		if (mp_parent) mp_parent->GetStateRegister().RequestUpdate();
	}

	bool StateRegister::IsModified() const
	{
		return m_modified;
	}
	bool StateRegister::RequiresUpdate() const
	{
		return m_requires_update;
	}

	void StateRegister::MakeUnmodified()
	{
		m_modified = false;
	}
	void StateRegister::Update()
	{
		m_requires_update = false;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



	// ~~~~~~~~ [CLASS] Updatable ~~~~~~~~
	Updatable::Updatable(Updatable&& other)
		: m_register(std::move(other.m_register))
	{}

	Updatable::Updatable(Updatable* parent)
		: m_register(parent)
	{}

	void Updatable::Update()
	{
		m_register.Update();
	}

	const StateRegister& Updatable::GetStateRegister() const
	{
		return m_register;
	}
	StateRegister& Updatable::GetStateRegister()
	{
		return m_register;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}