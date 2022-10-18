#ifndef BVH_H
#define BVH_H

#include "render_parts.hpp"
#include "object_container.hpp"
#include "rzexception.hpp"

#include <vector>
#include <numeric>
#include <iterator>

namespace RayZath::Engine
{
	template <class T, uint32_t leaf_size = 4u, uint32_t max_depth = 31u>
	class TreeNode
	{
	public:
		static_assert(leaf_size != 0);
		struct Children
		{
			enum class PartitionType : uint8_t
			{
				X = 2,
				Y = 1,
				Z = 0,
				Size = 3
			};

			TreeNode first, second;
			PartitionType type;

			Children(TreeNode first, TreeNode second, PartitionType type)
				: first(std::move(first))
				, second(std::move(second))
				, type(type)
			{}
		};

		using objects_t = std::vector<Handle<T>>;
		using objects_iterator = typename objects_t::iterator;

	private:
		std::unique_ptr<Children> m_children;
		objects_t m_objects;
		BoundingBox m_bb;

	public:
		TreeNode() = default;
		TreeNode(const BoundingBox& bb, objects_t& objects, const uint32_t depth = 0)
			: TreeNode(bb, objects.begin(), objects.end(), depth)
		{}
	private:
		TreeNode(const BoundingBox& bb, const objects_iterator begin, const objects_iterator end, const uint32_t depth)
			: m_bb(bb)
		{
			construct(begin, end, depth);
			fitBoundingBox();
		}

	public:
		const std::unique_ptr<Children>& children() const
		{
			return m_children;
		}
		const objects_t& objects() const
		{
			return m_objects;
		}

		const BoundingBox& boundingBox() const
		{
			return m_bb;
		}

		void clear()
		{
			m_children.release();
			m_objects.clear();
		}

		uint32_t treeSize() const
		{
			return children() ? children()->first.treeSize() + children()->second.treeSize() + 1 : 1;
		}
		bool isLeaf() const
		{
			return !m_children;
		}

	private:
		BoundingBox fitBoundingBox()
		{
			m_bb = BoundingBox();

			if (isLeaf())
			{
				if (!objects().empty())
				{
					m_bb = objects()[0]->boundingBox();
					for (size_t i = 1; i < objects().size(); i++)
					{
						m_bb.extendBy(objects()[i]->boundingBox());
					}
				}
			}
			else
			{
				if (children())
				{
					m_bb = children()->first.boundingBox();
					m_bb.extendBy(children()->second.boundingBox());
				}
			}

			return m_bb;
		}
		void construct(const objects_iterator begin, const objects_iterator end, const uint32_t depth = 0)
		{
			// all object can be stored as a tree leaf
			if (depth > max_depth || std::distance(begin, end) <= leaf_size ||
				(depth == 0 && std::distance(begin, end) <= 8))
			{
				m_objects = objects_t(std::make_move_iterator(begin), std::make_move_iterator(end));
				return;
			}

			// find all objects not suitable for further sub-partition (are too large)
			const auto node_size = boundingBox().max - boundingBox().min;
			auto size_split_point = std::partition(begin, end,
				[node_size](const auto& object)
				{
					const Math::vec3f object_size = object->boundingBox().max - object->boundingBox().min;
					return object_size.x < node_size.x&& object_size.y < node_size.y&& object_size.z < node_size.z;
				});
			const auto to_split_count = std::distance(begin, size_split_point);
			const auto too_large_count = std::distance(size_split_point, end);
			if (to_split_count != 0 && too_large_count != 0)
			{
				m_children = std::make_unique<Children>(
					TreeNode(m_bb, begin, size_split_point, depth + 1), // objects to split
					TreeNode(m_bb, size_split_point, end, depth + 1), // too large objects
					Children::PartitionType::Size);
				return;
			}
			else if (to_split_count == 0)
			{	// only too large objects left, so further sub-partition is ineffective
				m_objects = objects_t(std::make_move_iterator(size_split_point), std::make_move_iterator(end));
				return;
			}

			const auto to_split_begin = begin;
			const auto to_split_end = size_split_point;
			// find split point
			Math::vec3f split_point{};
			for (int32_t i = 0; i < to_split_count; i++)
				split_point += (to_split_begin[i]->boundingBox().centroid() - split_point) / float(i + 1);

			// count objects and compute distribution variance along each plane
			Math::vec3f variance_sum(0.0f);
			Math::vec3<uint32_t> split_count;
			for (int32_t i = 0; i < to_split_count; i++)
			{
				const auto diff = to_split_begin[i]->boundingBox().centroid() - split_point;
				variance_sum += diff * diff;
				split_count.x += uint32_t(to_split_begin[i]->boundingBox().centroid().x < split_point.x);
				split_count.y += uint32_t(to_split_begin[i]->boundingBox().centroid().y < split_point.y);
				split_count.z += uint32_t(to_split_begin[i]->boundingBox().centroid().z < split_point.z);
			}
			if (split_count == Math::vec3<uint32_t>(0))
			{	// no sub-partition is possible (all objects' centroids are in one single point)
				m_objects = objects_t(std::make_move_iterator(to_split_begin), std::make_move_iterator(to_split_end));
				return;
			}

			// score for each axis
			Math::vec3f	score = variance_sum / float(to_split_count);

			if (score.x >= score.y && score.x >= score.z && split_count.x)
			{	// split by X axis
				auto split_plane = std::partition(to_split_begin, to_split_end, [split_point](const auto& object)
					{ return object->boundingBox().centroid().x < split_point.x; });

				Math::vec3f max = m_bb.max, min = m_bb.min;
				max.x = min.x = split_point.x;
				m_children = std::make_unique<Children>(
					TreeNode(BoundingBox(m_bb.min, max), to_split_begin, split_plane, depth + 1),
					TreeNode(BoundingBox(min, m_bb.max), split_plane, to_split_end, depth + 1),
					Children::PartitionType::X);
				return;
			}
			else if (score.y >= score.x && score.y >= score.z && split_count.y)
			{	// split by Y axis
				auto split_plane = std::partition(to_split_begin, to_split_end, [split_point](const auto& object)
					{ return object->boundingBox().centroid().y < split_point.y; });

				Math::vec3f max = m_bb.max, min = m_bb.min;
				max.y = min.y = split_point.y;
				m_children = std::make_unique<Children>(
					TreeNode(BoundingBox(m_bb.min, max), to_split_begin, split_plane, depth + 1),
					TreeNode(BoundingBox(min, m_bb.max), split_plane, to_split_end, depth + 1),
					Children::PartitionType::Y);
			}
			else
			{	// split by Z axis
				auto split_plane = std::partition(to_split_begin, to_split_end, [split_point](const auto& object)
					{ return object->boundingBox().centroid().z < split_point.z; });

				Math::vec3f max = m_bb.max, min = m_bb.min;
				max.z = min.z = split_point.z;
				m_children = std::make_unique<Children>(
					TreeNode(BoundingBox(m_bb.min, max), to_split_begin, split_plane, depth + 1),
					TreeNode(BoundingBox(min, m_bb.max), split_plane, to_split_end, depth + 1),
					Children::PartitionType::Z);
			}
		}
	};

	template <class T>
	class ObjectContainerWithBVH
		: public ObjectContainer<T>
	{
	public:
		using objects_t = typename TreeNode<T>::objects_t;

	private:
		TreeNode<T> m_root;

	public:
		ObjectContainerWithBVH(Updatable* updatable)
			: ObjectContainer<T>(updatable)
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
