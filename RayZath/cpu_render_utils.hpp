#ifndef CPU_RENDER_UTILS
#define CPU_RENDER_UTILS

#include "vec2.h"
#include "vec3.h"
#include "color.h"

#include "mesh_component.hpp"

#include <limits>

namespace RayZath::Engine
{
	class Instance;
	class Triangle;
	struct Material;
}

namespace RayZath::Engine::CPU
{
	struct RNG
	{
	private:
		float a, b;

	public:
		RNG(const Math::vec2f32 seed, const float r);
		float unsignedUniform();
		float signedUniform();
	private:
		float fract(const float f);
	};

	struct Ray
	{
	public:
		Math::vec3f32 origin;
		Math::vec3f32 direction;

	public:
		Ray() = default;
		Ray(const Math::vec3f32 origin, const Math::vec3f32 direction)
			: origin(origin)
			, direction(direction)
		{
			this->direction.Normalize();
		}
	};
	struct RangedRay : public Ray
	{
	public:
		Math::vec2f32 near_far;

	public:
		RangedRay()
			: near_far(0.0f, std::numeric_limits<float>().max())
		{}
		RangedRay(
			const Math::vec3f32& origin,
			const Math::vec3f32& direction,
			const Math::vec2f32 near_far = Math::vec2f32(0.0f, std::numeric_limits<float>().max()))
			: Ray(origin, direction)
			, near_far(near_far)
		{}

	public:
		void resetRange(const Math::vec2f32 range = Math::vec2f32(0.0f, std::numeric_limits<float>().max()))
		{
			near_far = range;
		}
	};
	struct SceneRay : public RangedRay
	{
	public:
		const Material* material;
		Graphics::ColorF color;


	public:
		SceneRay()
			: RangedRay()
			, material(nullptr)
			, color(1.0f)
		{}
		SceneRay(
			const Math::vec3f32& origin,
			const Math::vec3f32& direction,
			const Material* material,
			const Graphics::ColorF& color = Graphics::ColorF(1.0f),
			const Math::vec2f32 near_far = Math::vec2f32(0.0f, std::numeric_limits<float>().max()))
			: RangedRay(origin, direction, near_far)
			, material(material)
			, color(color)
		{}
	};


	struct TraversalResult
	{
		const Instance* closest_instance = nullptr;
		const Triangle* closest_triangle = nullptr;
		Math::vec2f barycenter;
		bool external = true;
	};
	struct SurfaceProperties
	{
		const Material* surface_material;
		const Material* behind_material;

		Texcrd texcrd;
		Math::vec3f32 normal;
		Math::vec3f32 mapped_normal;

		Graphics::ColorF color;
		float metalness = 0.0f;
		float roughness = 0.0f;
		float emission = 0.0f;

		float fresnel = 1.0f;
		float reflectance = 0.0f;

		float tint_factor = 0.0f;
		Math::vec2f32 refraction_factors;

		SurfaceProperties(const Material* material)
			: surface_material(material)
			, behind_material(material)
		{}
	};
	struct TracingResult
	{
		Math::vec3f32 point;
		Math::vec3f32 next_direction;

		void repositionRay(SceneRay& ray) const
		{
			ray.origin = point;
			ray.direction = next_direction;
			ray.resetRange();
		}
	};


	// ~~~~~~~~ helper functions ~~~~~~~~
	Math::vec3f32 reflectVector(const Math::vec3f32& vI, const Math::vec3f32& vN);
	Math::vec3f32 halfwayVector(const Math::vec3f32& vI, const Math::vec3f32& vR);
	float rayToPointDistance(const Ray& ray, const Math::vec3f32& P);
	void rayPointCalculation(
		const Ray& ray, const Math::vec3f32& P,
		Math::vec3f32& vOP,
		float& dOP,
		float& vOP_dot_vD,
		float& dPQ);

	void localCoordinate(const Math::vec3f32& vN, Math::vec3f32& vX, Math::vec3f32& vY);

	Math::vec3f32 cosineSampleHemisphere(
		const float r1,
		const float r2,
		const Math::vec3f32& vN);
	Math::vec3f32 sampleSphere(
		const float r1,
		const float r2,
		const Math::vec3f32& vN);
	Math::vec3f32 sampleHemisphere(
		const float r1,
		const float r2,
		const Math::vec3f32& vN);
	Math::vec3f32 sampleDisk(
		const float r1,
		const float r2,
		const Math::vec3f32& vN,
		const float radius);

	// returns probability of reflection
	float fresnelSpecularRatio(
		const Math::vec3f32& vN,
		const Math::vec3f32& vI,
		const float n1,
		const float n2,
		Math::vec2f32& factors);
	float fresnelSpecularRatioSchlick(
		const Math::vec3f32& vN,
		const Math::vec3f32& vI,
		const float n1,
		const float n2);
}

#endif
