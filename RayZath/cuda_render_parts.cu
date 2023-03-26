#include "cuda_render_parts.cuh"
#include "cuda_world.cuh"

#include "mesh_component.hpp"
#include "render_parts.hpp"

namespace RayZath::Cuda
{
	Triangle::Triangle(const RayZath::Engine::Triangle& hostTriangle)
	{
		m_normal = hostTriangle.normal;
		m_material_id = hostTriangle.material_id & 0x3F;
	}

	void Triangle::setVertices(const vec3f& v1, const vec3f& v2, const vec3f& v3)
	{
		m_v1 = v1;
		m_v2 = v2;
		m_v3 = v3;
	}
	void Triangle::setTexcrds(const vec2f& t1, const vec2f& t2, const vec2f& t3)
	{
		m_t1 = t1;
		m_t2 = t2;
		m_t3 = t3;
	}
	void Triangle::setNormals(const vec3f& n1, const vec3f& n2, const vec3f& n3)
	{
		m_n1 = n1;
		m_n2 = n2;
		m_n3 = n3;
	}

	CoordSystem::CoordSystem(const Math::vec3f& x, const Math::vec3f& y, const Math::vec3f& z)
		: x_axis(x)
		, y_axis(y)
		, z_axis(z)
	{}
	CoordSystem& CoordSystem::operator=(const RayZath::Engine::CoordSystem& coord_system)
	{
		x_axis = coord_system.xAxis();
		y_axis = coord_system.yAxis();
		z_axis = coord_system.zAxis();
		return *this;
	}

	Transformation& Transformation::operator=(const RayZath::Engine::Transformation& t)
	{
		position = t.position();
		scale = t.scale();
		coord_system = t.coordSystem();
		return *this;
	}

	BoundingBox::BoundingBox()
		: min(0.0f)
		, max(0.0f)
	{}
	BoundingBox::BoundingBox(const RayZath::Engine::BoundingBox& box)
		: min(box.min)
		, max(box.max)
	{}

	BoundingBox& BoundingBox::operator=(const RayZath::Engine::BoundingBox& box)
	{
		this->min = box.min;
		this->max = box.max;
		return *this;
	}
}
