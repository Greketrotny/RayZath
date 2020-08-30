#ifndef UPDATABLE_H
#define UPDATABLE_H

namespace RayZath
{
	class Updatable
	{
	private:
		Updatable* mp_parent;
		bool m_requires_update;


	protected:
		Updatable(Updatable* updatable_parent);
		~Updatable();


	public:
		Updatable* GetUpdatableParent();
		const Updatable* GetUpdatableParent() const;
		bool RequiresUpdate() const;
		void RequestUpdate();
		virtual void Update();
		void Updated();
	};
}

#endif // !UPDATABLE_H