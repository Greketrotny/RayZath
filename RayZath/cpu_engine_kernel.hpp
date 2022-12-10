#ifndef CPU_ENGINE_KERNEL_HPP
#define CPU_ENGINE_KERNEL_HPP

#include "world.hpp"

#include "cpu_render_utils.hpp"

namespace RayZath::Engine::CPU
{
	class Kernel
	{
	private:
		const World* mp_world = nullptr;


	public:
		void setWorld(World& world);
		Graphics::ColorF render(const Camera& camera, const Math::vec2ui32 pixel) const;


	private:
		void generateCameraRay(const Camera& camera, RangedRay& ray, const Math::vec2ui32& pixel) const;

		bool closestIntersection(SceneRay& ray) const;
	};
}

#endif 
