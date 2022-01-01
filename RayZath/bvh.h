#ifndef BVH_H
#define BVH_H

#include "render_parts.h"
#include "object_container.h"

#include <vector>

namespace RayZath::Engine
{
	template <class T> struct TreeNode
	{
	private:
		static constexpr uint32_t s_leaf_size = 16u;
		TreeNode* m_child[8];
		std::vector<Handle<T>> objects;
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
			const Handle<T>& object,
			uint32_t depth)
		{
			if (m_is_leaf)
			{
				if (depth > 8u || objects.size() < s_leaf_size)
				{	// insert the object into leaf

					objects.push_back(object);
				}
				else
				{	// turn leaf into node and reinsert

					m_is_leaf = false;

					// copy objects to temporal storage
					std::vector<Handle<T>> node_objects = objects;
					// add new object to storage
					node_objects.push_back(object);
					objects.clear();

					// distribute objects into child nodes
					for (uint32_t i = 0u; i < node_objects.size(); i++)
					{
						// find child id for the object
						Math::vec3f vCP =
							node_objects[i]->GetBoundingBox().GetCentroid() -
							m_bb.GetCentroid();

						int child_id = 0;
						Math::vec3f child_extent = m_bb.max;
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
				Math::vec3f vCP =
					object->GetBoundingBox().GetCentroid() -
					m_bb.GetCentroid();

				int child_id = 0;
				Math::vec3f child_extent = m_bb.max;
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
		bool Remove(const Handle<T>& object)
		{
			if (m_is_leaf)
			{
				for (uint32_t i = 0; i < objects.size(); ++i)
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
						if (m_child[i]->Remove(object))
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
				for (auto& o : objects)
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

		TreeNode* GetChild(uint32_t child_id)
		{
			return m_child[child_id];
		}
		const TreeNode* GetChild(uint32_t child_id) const
		{
			return m_child[child_id];
		}
		uint32_t GetChildCount() const
		{
			uint32_t child_count = 0u;
			for (uint32_t i = 0u; i < 8u; i++)
				if (m_child[i] != nullptr)
					child_count++;

			return child_count;
		}
		uint32_t GetRecursiveChildCount() const
		{
			unsigned int child_count = 0u;
			for (int i = 0; i < 8; i++)
			{
				if (m_child[i])
				{
					child_count += m_child[i]->GetRecursiveChildCount() + 1u;
				}
			}
			return child_count;
		}

		const Handle<T>& GetObject(uint32_t object_index) const
		{
			return objects[object_index];
		}
		uint32_t GetObjectCount() const
		{
			return uint32_t(objects.size());
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
		bool Insert(const Handle<T>& object)
		{
			return m_root.Insert(object);
		}
		bool Remove(const Handle<T>& object)
		{
			return m_root.Remove(object);
		}
		void Construct(const ObjectContainer<T>& objects)
		{
			Reset();

			for (uint32_t i = 0u; i < objects.GetCount(); i++)
				m_root.ExtendBoundingBox(objects[i]->GetBoundingBox());

			for (uint32_t i = 0u; i < objects.GetCount(); i++)
				m_root.Insert(objects[i], 0u);

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
		uint32_t GetTreeSize()
		{
			return 1u + m_root.GetRecursiveChildCount();
		}
	};


	template <class T>
	struct ObjectContainerWithBVH
		: public ObjectContainer<T>
	{
	private:
		BVH<T> m_bvh;


	public:
		ObjectContainerWithBVH(
			Updatable* updatable)
			: ObjectContainer<T>(updatable)
		{}


	public:
		bool Destroy(const Handle<T>& object)
		{
			bool bvh_result = m_bvh.Remove(object);
			bool cont_result = ObjectContainer<T>::Destroy(object);
			this->GetStateRegister().RequestUpdate();
			return (bvh_result && cont_result);
		}
		bool Destroy(const uint32_t& index)
		{
			if (index >= this->GetCount())
				return false;

			this->GetStateRegister().RequestUpdate();
			return (m_bvh.Remove((*this)[index]) &&
				ObjectContainer<T>::Destroy(index));
		}
		void DestroyAll()
		{
			ObjectContainer<T>::DestroyAll();
			m_bvh.Reset();
			this->GetStateRegister().RequestUpdate();
		}
	public:
		void Update() override
		{
			if (!ObjectContainer<T>::GetStateRegister().RequiresUpdate()) return;

			ObjectContainer<T>::Update();
			m_bvh.Construct(*this);

			ObjectContainer<T>::GetStateRegister().Update();
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