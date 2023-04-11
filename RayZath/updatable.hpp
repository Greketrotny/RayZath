#ifndef UPDATABLE_H
#define UPDATABLE_H

#include <stdint.h>

namespace RayZath::Engine
{
	class Updatable;

	struct StateRegister
	{
	private:
		Updatable* mp_parent;
		bool m_modified, m_requires_update;


	public:
		StateRegister(const StateRegister& other);
		StateRegister(StateRegister&& other) noexcept;
		StateRegister(Updatable* parent);


	public:
		void MakeModified();
		void RequestUpdate();

		bool IsModified() const;
		bool RequiresUpdate() const;

		void MakeUnmodified();
		void update();
	};

	class Updatable
	{
	private:
		StateRegister m_register;


	protected:
		Updatable(Updatable* updatable_parent);
		virtual ~Updatable() = default;


	public:
		virtual void update();
		const StateRegister& stateRegister() const;
		StateRegister& stateRegister();


		friend struct StateRegister;
	};
}

#endif // !UPDATABLE_H