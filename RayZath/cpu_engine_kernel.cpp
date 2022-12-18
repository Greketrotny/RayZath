#include "cpu_engine_kernel.hpp"

namespace RayZath::Engine::CPU
{
	void Kernel::setWorld(World& world)
	{
		mp_world = &world;
	}

	Graphics::ColorF Kernel::render(
		const Camera& camera,
		const Math::vec2ui32 pixel) const
	{
		RZAssertCore(bool(mp_world), "mp_world was nullptr");

		SceneRay ray{};
		generateCameraRay(camera, ray, pixel);

		auto hit = closestIntersection(ray);

		return Graphics::ColorF(
			hit,
			hit,
			hit,
			1.0f);
	}

	void Kernel::generateCameraRay(const Camera& camera, RangedRay& ray, const Math::vec2ui32& pixel) const
	{
		using namespace Math;
		ray.origin = vec3f32(0.0f);

		// ray to screen deflection
		const float tana = std::tanf(camera.fov().value() * 0.5f);
		const vec2f32 dir =
			(((vec2f32(pixel) + vec2f32(0.5f)) /
				vec2f32(camera.resolution())) -
				vec2f32(0.5f)) *
			vec2f(tana, -tana / camera.aspectRatio());
		ray.direction.x = dir.x;
		ray.direction.y = dir.y;
		ray.direction.z = 1.0f;

		// camera transformation
		ray.origin = camera.coordSystem().transformForward(ray.origin);
		ray.direction = camera.coordSystem().transformForward(ray.direction);
		ray.direction.Normalize();
		ray.origin += camera.position();

		// apply near/far clipping plane
		ray.near_far = camera.nearFar();
	}

	bool Kernel::closestIntersection(SceneRay& ray) const
	{
		const auto& instances = mp_world->container<World::ObjectType::Instance>();
		if (instances.empty()) return false;
		if (!instances.root().boundingBox().rayIntersection(ray)) return false;
		return traverseWorld(instances.root(), ray);
	}
	bool Kernel::closestIntersection(const Instance& instance, SceneRay& ray) const
	{
		if (!instance.boundingBox().rayIntersection(ray)) return false;

		TraversalResult traversal;
		RangedRay local_ray = ray;
		instance.transformation().transformG2L(local_ray);

		const float length_factor = local_ray.direction.Magnitude();
		local_ray.near_far *= length_factor;
		local_ray.direction.Normalize();

		const auto& mesh = instance.mesh();
		if (!mesh) return false;
		const auto* const closest_triangle = traversal.closest_triangle;
		traversal.closest_triangle = nullptr;
		
		closestIntersection(*mesh, local_ray, traversal);
		if (traversal.closest_triangle)
		{
			traversal.closest_instance = &instance;
			ray.near_far = local_ray.near_far / length_factor;
			return true;
		}
		else
		{
			traversal.closest_triangle = closest_triangle;
			return false;
		}
	}
	void Kernel::closestIntersection(const Mesh& mesh, RangedRay& ray, TraversalResult& traversal) const
	{
		auto traverse = [&](const auto& self, const auto& node) {
			if (!node.boundingBox().rayIntersection(ray)) return;

			if (node.isLeaf())
			{
				for (const auto& triangle : node.objects())
				{
					triangle->closestIntersection(ray, traversal, mesh);
				}
			}
			else
			{
				RZAssertCore(node.children(), "If not leaf, should have children.");
				self(self, node.children()->first);
				self(self, node.children()->second);
			}
		};

		traverse(traverse, mesh.triangles().getBVH().rootNode());
	}

	bool Kernel::traverseWorld(const tree_node_t& node, SceneRay& ray) const
	{
		if (node.isLeaf())
		{
			for (const auto& object : node.objects())
			{
				if (!object) continue;
				if (closestIntersection(*object, ray))
				{
					return true;
				}
			}
		}
		else
		{
			const auto& first_child = node.children()->first;
			if (first_child.boundingBox().rayIntersection(ray))
			{
				if (traverseWorld(first_child, ray))
					return true;
			}
			const auto& second_child = node.children()->second;
			if (second_child.boundingBox().rayIntersection(ray))
			{
				if (traverseWorld(second_child, ray))
					return true;
			}
		}

		return false;
	}
}
