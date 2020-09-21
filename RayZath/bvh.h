#ifndef BVH_H
#define BVH_H

#include "render_parts.h"
#include "object_container.h"

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
				if (depth > 4u || objects.size() < s_leaf_size)
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
						Math::vec3<float> child_extent = m_bb.max;
						if (vCP.x < 0.0f)
						{
							child_id += 4;
							child_extent.x = m_bb.min.x;
						}
						if (vCP.y < 0.0f)
						{
							child_id += 2;
							child_extent.y = m_bb.min.y;
						}
						if (vCP.z < 0.0f)
						{
							child_id += 1;
							child_extent.z = m_bb.min.z;
						}

						// insert object to the corresponding child node
						if (!m_child[child_id]) m_child[child_id] = new TreeNode<T>(BoundingBox(
							m_bb.GetCentroid(), child_extent));
						m_child[child_id]->Insert(node_objects[i], depth + 1);
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
				Math::vec3<float> child_extent = m_bb.max;
				if (vCP.x < 0.0f)
				{
					child_id += 4;
					child_extent.x = m_bb.min.x;
				}
				if (vCP.y < 0.0f)
				{
					child_id += 2;
					child_extent.y = m_bb.min.y;
				}
				if (vCP.z < 0.0f)
				{
					child_id += 1;
					child_extent.z = m_bb.min.z;
				}

				// insert object to the corresponding child node
				if (!m_child[child_id]) m_child[child_id] = new TreeNode<T>(BoundingBox(
					m_bb.GetCentroid(), child_extent));
				m_child[child_id]->Insert(object, depth + 1);
			}

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
		void ExtendBoundingBox(const BoundingBox& bb)
		{
			m_bb.ExtendBy(bb);
		}

		TreeNode* const GetChild(unsigned int child_id)
		{
			return m_child[child_id];
		}
		const TreeNode* const GetChild(unsigned int child_id) const
		{
			return m_child[child_id];
		}
		unsigned int GetChildCount() const
		{
			unsigned int child_count = 0u;
			for (int i = 0; i < 8; i++)
			{
				if (m_child[i])
				{
					child_count += m_child[i]->GetChildCount() + 1u;
				}
			}
			return child_count;
		}
		
		const T* GetObject(unsigned int object_index) const
		{
			return objects[object_index];
		}
		unsigned int GetObjectCount() const
		{
			return objects.size();
		}

		BoundingBox GetBoundingBox() const
		{
			return m_bb;
		}

		bool IsLeaf() const
		{
			return m_is_leaf;
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
		void Construct(const ObjectContainer<T>& objects)
		{
			Reset();

			unsigned int index = 0u, count = 0u;

			// Set BB of root node to BB of the first object
			while (index < objects.GetCapacity())
			{
				if (objects[index])
				{
					m_root.SetBoundingBox(objects[index]->GetBoundingBox());
					count++;

					break;
				}
				index++;
			}

			// Expand root BB by BBs of all objects
			while (index < objects.GetCapacity() && count < objects.GetCount())
			{
				if (objects[index])
				{
					m_root.ExtendBoundingBox(objects[index]->GetBoundingBox());
					count++;
				}
				index++;
			}

			// Insert all objects into BVH
			index = 0u;
			count = 0u;
			while (index < objects.GetCapacity() && count < objects.GetCount())
			{
				if (objects[index])
				{
					m_root.Insert(objects[index]);
					count++;
				}
				index++;
			}

			m_root.FitBoundingBox();
		}
		void Reset()
		{
			m_root.Reset();
		}
		void FitBoundingBox()
		{
			m_root.FitBoundingBox();
		}

		const TreeNode<T>& GetRootNode()
		{
			return m_root;
		}
		unsigned int GetTreeSize()
		{
			return 1u + m_root.GetChildCount();
		}
	};



	template <class T> struct ObjectContainerWithBVH
		: public ObjectContainer<T>
	{
	private:
		BVH<T> m_bvh;


	public:
		ObjectContainerWithBVH(Updatable* updatable, size_t capacity = 16u)
			: ObjectContainer<T>(updatable, capacity)
		{}
		~ObjectContainerWithBVH()
		{}


	public:
		T* operator[](size_t index)
		{
			return (*static_cast<ObjectContainer<T>*>(this))[index];
		}
		const T* operator[](size_t index) const
		{
			return (*static_cast<const ObjectContainer<T>*>(this))[index];
		}


	public:
		T* CreateObject(const ConStruct<T>& con_struct)
		{
			T* object = ObjectContainer<T>::CreateObject(con_struct);
			//m_bvh.Insert(object);
			//m_bvh.FitBoundingBox();
			return object;
		}
		bool DestroyObject(const T* object)
		{
			bool bvh_result = m_bvh.Remove(object);
			bool cont_result = ObjectContainer<T>::DestroyObject(object);
			return (bvh_result && cont_result);
		}
		void DestroyAllObjects()
		{
			ObjectContainer<T>::DestroyAllObjects();
			m_bvh.Reset();
		}
	public:
		void Update() override
		{
			m_bvh.Construct(*this);
		}

	public:
		const BVH<T>& GetBVH() const
		{
			return m_bvh;
		}
		BVH<T>& GetBVH()
		{
			return m_bvh;
		}
	};
}

#endif