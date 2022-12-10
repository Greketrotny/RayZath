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
		return instances.root().boundingBox().rayIntersection(ray);
	}
}
