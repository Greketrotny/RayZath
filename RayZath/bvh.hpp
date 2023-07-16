#ifndef BVH_H
#define BVH_H

#include "bvh_tree_node.hpp"
#include "resource_container.hpp"

namespace RayZath::Engine
{
	template <class T>
	class ObjectContainerWithBVH
		: public ResourceContainer<T>
	{
	public:
		using tree_node_t = TreeNode<T>;
		using objects_t = typename tree_node_t::objects_t;

	private:
		tree_node_t m_root;

	public:
		ObjectContainerWithBVH(Updatable* updatable)
			: ResourceContainer<T>(updatable)
		{}

		const TreeNode<T>& root() const
		{
			return m_root;
		}

		void update() override
		{
			if (!ObjectContainer<T>::stateRegister().RequiresUpdate()) return;

			// update objects
			ObjectContainer<T>::update();

			// construct bvh
			BoundingBox bb;
			objects_t objects;
			if (!this->empty())
				bb = this->operator[](0)->boundingBox();
			for (uint32_t i = 0; i < this->count(); i++)
			{
				auto object = this->operator[](i);
				if (object->mesh())
				{
					bb.extendBy(object->boundingBox());
					objects.push_back(std::move(object));
				}
			}
			m_root = TreeNode<T>(bb, objects);

			ObjectContainer<T>::stateRegister().update();
		}
	};
}

#endif
