#include "updatable.hpp"

#include <memory>

namespace RayZath::Engine
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
		if (mp_parent) mp_parent->stateRegister().MakeModified();
	}
	void StateRegister::RequestUpdate()
	{
		m_requires_update = true;
		m_modified = true;
		if (mp_parent) mp_parent->stateRegister().RequestUpdate();
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
	void StateRegister::update()
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

	void Updatable::update()
	{
		m_register.update();
	}

	const StateRegister& Updatable::stateRegister() const
	{
		return m_register;
	}
	StateRegister& Updatable::stateRegister()
	{
		return m_register;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}