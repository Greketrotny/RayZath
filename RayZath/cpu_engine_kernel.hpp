#ifndef CPU_ENGINE_KERNEL_HPP
#define CPU_ENGINE_KERNEL_HPP

#include "world.hpp"

#include "cpu_render_utils.hpp"

#include <type_traits>

namespace RayZath::Engine::CPU
{
	class Kernel
	{
	private:
		const World* mp_world = nullptr;

		using tree_node_t = std::decay_t<decltype(mp_world->container<World::ObjectType::Instance>())>::tree_node_t;

	public:
		void setWorld(World& world);
		Graphics::ColorF render(const Camera& camera, const Math::vec2ui32 pixel) const;


	private:
		void generateCameraRay(const Camera& camera, RangedRay& ray, const Math::vec2ui32& pixel) const;


		bool closestIntersection(SceneRay& ray, SurfaceProperties& surface) const;
		void closestIntersection(const Instance& instance, SceneRay& ray, TraversalResult& traversal) const;
		void closestIntersection(const Mesh& mesh, RangedRay& ray, TraversalResult& traversal) const;

		void traverseWorld(const tree_node_t& node, SceneRay& ray, TraversalResult& traversal) const;
		void analyzeIntersection(
			const Instance& instance, 
			TraversalResult& traversal, 
			SurfaceProperties& surface) const;
	};
}

#endif 
