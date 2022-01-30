#include "cuda_render_parts.cuh"
#include "rzexception.h"
#include "cuda_world.cuh"

namespace RayZath::Cuda
{
	Triangle::Triangle(const RayZath::Engine::Triangle& hostTriangle)
	{
		m_normal = hostTriangle.normal;
		m_material_id = hostTriangle.material_id & 0x3F;
	}

	void Triangle::SetVertices(const vec3f& v1, const vec3f& v2, const vec3f& v3)
	{
		m_v1 = v1;
		m_v2 = v2;
		m_v3 = v3;
	}
	void Triangle::SetTexcrds(const vec2f& t1, const vec2f& t2, const vec2f& t3)
	{
		m_t1 = t1;
		m_t2 = t2;
		m_t3 = t3;
	}
	void Triangle::SetNormals(const vec3f& n1, const vec3f& n2, const vec3f& n3)
	{
		m_n1 = n1;
		m_n2 = n2;
		m_n3 = n3;
	}
}