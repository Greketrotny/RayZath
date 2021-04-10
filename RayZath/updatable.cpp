#include "updatable.h"

namespace RayZath
{
	// ~~~~~~~~ [STRUCT] StateRegister ~~~~~~~~
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