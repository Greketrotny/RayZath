#include "cpu_render_utils.hpp"

#include <numbers>
#include <cmath>

namespace RayZath::Engine::CPU
{
	RNG::RNG(const Math::vec2f32 seed, const float r)
		: a(seed.x + seed.y)
		, b(r * 245.310913f)
	{}
	float RNG::unsignedUniform()
	{
		const float af = (a + 0.2311362f) * (b + 13.054377f);
		const float bf = (a + 251.78431f) + (b - 73.054312f);
		a = af - float(int32_t(af));
		b = bf - float(int32_t(bf));
		return std::fabsf(b);
	}
	float RNG::signedUniform()
	{
		return unsignedUniform() * 2.0f - 1.0f;
	}
	float RNG::fract(const float f)
	{
		return f - truncf(f);
	}

	Math::vec3f32 reflectVector(const Math::vec3f32& vI, const Math::vec3f32& vN)
	{
		return (vN * -2.0f * Math::vec3f32::DotProduct(vN, vI) + vI);
	}
	Math::vec3f32 halfwayVector(const Math::vec3f32& vI, const Math::vec3f32& vR)
	{
		return ((-vI) + vR).Normalized();
	}
	float rayToPointDistance(const Ray& ray, const Math::vec3f32& P)
	{
		// O - ray origin
		// P - specified point
		// vD - ray direction

		const Math::vec3f32 vOP = P - ray.origin;
		const float dOP = vOP.Magnitude();
		const float vOP_dot_vD = Math::vec3f32::DotProduct(vOP, ray.direction);
		return std::sqrtf(dOP * dOP - vOP_dot_vD * vOP_dot_vD);
	}
	void rayPointCalculation(
		const Ray& ray, const Math::vec3f32& P,
		Math::vec3f32& vOP,
		float& dOP,
		float& vOP_dot_vD,
		float& dPQ)
	{
		/*
			^
			|
			Q ---- P
			|     /
			|    /	    // O - ray origin
			|   /	    // P - specified point
			|  /	    // Q - closest point to P lying on ray
			| /
			|/
			O
		*/

		vOP = P - ray.origin;
		dOP = vOP.Magnitude();
		vOP_dot_vD = Math::vec3f32::DotProduct(vOP, ray.direction);
		dPQ = std::sqrtf(dOP * dOP - vOP_dot_vD * vOP_dot_vD);
	}

	void localCoordinate(const Math::vec3f32& vN, Math::vec3f32& vX, Math::vec3f32& vY)
	{
		bool b = (std::fabsf(vN.x) > std::fabsf(vN.y));
		vX.x = static_cast<float>(!b);
		vX.y = static_cast<float>(b);
		vX.z = 0.0f;

		vY = Math::vec3f32::CrossProduct(vN, vX);
		vX = Math::vec3f32::CrossProduct(vN, vY);
	}

	Math::vec3f32 cosineSampleHemisphere(
		const float r1,
		const float r2,
		const Math::vec3f32& vN)
	{
		// create local coordinate space vectors
		Math::vec3f32 vX, vY;
		localCoordinate(vN, vX, vY);

		const float phi = r1 * 6.283185f;
		const float theta = r2;

		// calculate sample direction
		const float sqrt_theta = std::sqrtf(theta);
		return vX * sqrt_theta * std::cosf(phi) + vY * sqrt_theta * std::sinf(phi) + vN * std::sqrtf(1.0f - theta);
		//				  along local x axis		+ along local z axis		+ along normal
	}
	Math::vec3f32 sampleSphere(
		const float r1,
		const float r2,
		const Math::vec3f32& vN)
	{
		// create local coordinate space vectors
		Math::vec3f32 vX, vY;
		localCoordinate(vN, vX, vY);

		// calculate phi and theta angles
		const float phi = r1 * 6.283185f;
		const float theta = std::acosf(1.0f - 2.0f * r2);

		// calculate sample direction
		const float sin_theta = std::sinf(theta);
		return vX * sin_theta * std::cosf(phi) + vY * sin_theta * std::sinf(phi) + vN * std::cosf(theta);
		//		along local x axis			+ along local y axis			+ along normal
	}
	Math::vec3f32 sampleHemisphere(
		const float r1,
		const float r2,
		const Math::vec3f32& vN)
	{
		return sampleSphere(r1, r2 * 0.5f, vN);
	}
	Math::vec3f32 sampleDisk(
		const float r1,
		const float r2,
		const Math::vec3f32& vN,
		const float radius)
	{
		Math::vec3f32 vX, vY;
		localCoordinate(vN, vX, vY);
		const float phi = r1 * 2.0f * std::numbers::pi_v<float>;
		const float mag = std::sqrtf(r2);
		return (vX * std::sinf(phi) + vY * std::cosf(phi)) * mag * radius;
	}

	// returns probability of reflection
	float fresnelSpecularRatio(
		const Math::vec3f32& vN,
		const Math::vec3f32& vI,
		const float n1,
		const float n2,
		Math::vec2f32& factors)
	{
		const float ratio = n1 / n2;
		const float cosi = std::fabsf(Math::vec3f32::DotProduct(vI, vN));
		const float sin2_t = ratio * ratio * (1.0f - cosi * cosi);
		if (sin2_t >= 1.0f) return 1.0f; // total 'internal' reflection

		const float cost = std::sqrtf(1.0f - sin2_t);
		const float Rp = ((n1 * cosi) - (n2 * cost)) / ((n1 * cosi) + (n2 * cost));
		const float Rs = ((n2 * cosi) - (n1 * cost)) / ((n2 * cosi) + (n1 * cost));

		factors = Math::vec2f32(ratio, ratio * cosi - cost);
		return (Rs * Rs + Rp * Rp) / 2.0f;
	}
	float fresnelSpecularRatioSchlick(
		const Math::vec3f32& vN,
		const Math::vec3f32& vI,
		const float n1,
		const float n2)
	{
		const float ratio = (n1 - n2) / (n1 + n2);
		const float r0 = ratio * ratio;
		const float vN_dot_vI = fabsf(Math::vec3f32::DotProduct(vN, vI));
		return r0 + (1.0f - r0) * std::powf(1.0f - vN_dot_vI, 5.0f);
	}
}
