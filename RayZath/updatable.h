#ifndef UPDATABLE_H
#define UPDATABLE_H

#include <stdint.h>

namespace RayZath
{
	class Updatable;

	struct StateRegister
	{
	private:
		Updatable* mp_parent;
		bool m_modified, m_requires_update;


	public:
		StateRegister(const StateRegister& other) = delete;
		StateRegister(StateRegister&& other);
		StateRegister(Updatable* parent);


	public:
		void MakeModified();
		void RequestUpdate();

		bool IsModified() const;
		bool RequiresUpdate() const;

		void MakeUnmodified();
		void Update();
	};

	class Updatable
	{
	private:
		StateRegister m_register;


	protected:
		Updatable(const Updatable& other) = delete;
		Updatable(Updatable&& other);
		Updatable(Updatable* updatable_parent);
		virtual ~Updatable() = default;


	public:
		virtual void Update();
		const StateRegister& GetStateRegister() const;
		StateRegister& GetStateRegister();


		friend struct StateRegister;
	};
}

#endif // !UPDATABLE_H