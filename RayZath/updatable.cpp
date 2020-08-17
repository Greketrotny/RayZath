#include "updatable.h"

namespace RayZath
{
	Updatable::Updatable(Updatable* updatable_parent)
		: mp_parent(updatable_parent)
		, m_requires_update(true)
	{
	}
	Updatable::~Updatable()
	{
	}

	Updatable* Updatable::GetUpdatableParent()
	{
		return mp_parent;
	}
	bool Updatable::RequiresUpdate()
	{
		return m_requires_update;
	}
	void Updatable::RequestUpdate()
	{
		m_requires_update = true;
		if (mp_parent) mp_parent->RequestUpdate();
	}
	void Updatable::Updated()
	{
		m_requires_update = false;
	}
}