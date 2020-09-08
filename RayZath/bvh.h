#ifndef BVH_H
#define BVH_H

#include "render_parts.h"
#include <vector>

namespace RayZath
{
	template <class T> struct TreeNode
	{
	private:
		static constexpr unsigned int s_leaf_size = 4u;
		TreeNode* m_child[8];
		std::vector<const T*> objects;
		BoundingBox m_bb;
		bool m_is_leaf;


	public:
		TreeNode(BoundingBox bb = BoundingBox())
			: m_bb(bb)
			, m_is_leaf(true)
		{
			for (int i = 0; i < 8; i++)
				m_child[i] = nullptr;
		}
		~TreeNode()
		{
			for (int i = 0; i < 8; i++)
			{
				if (m_child[i]) delete m_child[i];
				m_child[i] = nullptr;
			}

			objects.clear();
			m_is_leaf = true;
		}


	public:
		bool Insert(
			const T* object,
			unsigned int depth = 0u)
		{
			if (m_is_leaf)
			{
				if (depth > 16u || objects.size() < s_leaf_size)
				{	// insert the object into leaf

					objects.push_back(object);
				}
				else
				{	// turn leaf into node and reinsert

					m_is_leaf = false;

					// copy objects to temporal storage
					std::vector<const T*> node_objects = objects;
					// add new object to storage
					node_objects.push_back(object);
					objects.clear();

					// distribute objects into child nodes
					for (size_t i = 0u; i < node_objects.size(); i++)
					{
						// find child id for the object
						Math::vec3<float> vCP =
							node_objects[i]->GetBoundingBox().GetCentroid() -
							m_bb.GetCentroid();

						int child_id = 0;
						if (vCP.x < 0.0f) child_id += 4;
						if (vCP.y < 0.0f) child_id += 2;
						if (vCP.z < 0.0f) child_id += 1;

						// insert object to the corresponding child node
						if (m_child[child_id]) m_child[child_id]->Insert(object, depth + 1);
					}
				}
			}
			else
			{
				// find child id for the object
				Math::vec3<float> vCP =
					object->GetBoundingBox().GetCentroid() -
					m_bb.GetCentroid();

				int child_id = 0;
				if (vCP.x < 0.0f) child_id += 4;
				if (vCP.y < 0.0f) child_id += 2;
				if (vCP.z < 0.0f) child_id += 1;

				// insert object to the corresponding child node
				if (m_child[child_id]) m_child[child_id]->Insert(object, depth + 1);
			}

			FitBoundingBox();
			return true;
		}
		bool Remove(const T* object)
		{
			if (m_is_leaf)
			{
				for (size_t i = 0; i < objects.size(); ++i)
				{
					if (object == objects[i])
					{
						objects.erase(objects.begin() + i);
						return true;
					}
				}
			}
			else
			{
				for (int i = 0; i < 8; i++)
				{
					if (m_child[i])
					{
						if (m_child[i].Remove(object))
							return true;
					}
				}
			}

			return false;
		}
		BoundingBox FitBoundingBox()
		{
			if (objects.size() > 0u)
			{
				m_bb = objects[0]->GetBoundingBox();
				for (auto* o : objects)
				{
					m_bb.ExtendBy(o->GetBoundingBox());
				}
				return m_bb;
			}
			else
			{
				int i = 0;

				while (i < 8)
				{
					if (m_child[i])
					{
						m_bb = m_child[i]->FitBoundingBox();
						i++;
						break;
					}
					i++;
				}
				while (i < 8)
				{
					if (m_child[i])
						m_bb.ExtendBy(m_child[i]->FitBoundingBox());

					i++;
				}
				return m_bb;
			}
		}
		void Reset()
		{
			for (int i = 0; i < 8; i++)
			{
				if (m_child[i]) delete m_child[i];
				m_child[i] = nullptr;
			}

			objects.clear();
			m_is_leaf = true;
		}

		void SetBoundingBox(const BoundingBox& bb)
		{
			m_bb = bb;
		}

		BoundingBox GetBoundingBox()
		{
			return m_bb;
		}
		TreeNode* const GetChild(unsigned int child_id)
		{
			return m_child[child_id];
		}
	};

	template <class T> class BVH
	{
	private:
		TreeNode<T> m_root;


	public:
		BVH()
		{}
		~BVH()
		{}


	public:
		bool Insert(const T* object)
		{
			return m_root.Insert(object);
		}
		bool Remove(const T* object)
		{
			return m_root.Remove(object);
		}
		/*void Construct(const ObjectContainer<T>& objects)
		{
			Reset();

			unsigned int index = 0u, count = 0u;
			while (index < objects.GetCapacity())
			{
				if (objects[index])
				{
					count++;

					m_root.SetBoundingBox(objects[index]->GetBoundingBox());
					break;
				}
				index++;
			}

			while (index < objects.GetCapacity() && count < objects.GetCount())
			{
				if (objects[index])
				{
					m_root.Insert(objects[index]);
				}
			}

			m_root.FitBoundingBox();
		}*/
		void Reset()
		{
			m_root.Reset();
		}
	};
}

#endif