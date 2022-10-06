#include "world.hpp"

#include "rzexception.hpp"

#include "loader.hpp"
#include "json_loader.hpp"
#include "saver.hpp"
#include "json_saver.hpp"

#include <string_view>
#include <fstream>
#include <sstream>


namespace RayZath::Engine
{
	// ~~~~~~~~ [CLASS] World ~~~~~~~~
	World::World()
		: Updatable(nullptr)
		, m_containers(
			ObjectContainer<Texture>(this),
			ObjectContainer<NormalMap>(this),
			ObjectContainer<MetalnessMap>(this),
			ObjectContainer<RoughnessMap>(this),
			ObjectContainer<EmissionMap>(this),
			ObjectContainer<Material>(this),
			ObjectContainer<MeshStructure>(this),
			ObjectContainer<Camera>(nullptr),
			ObjectContainer<SpotLight>(this),
			ObjectContainer<DirectLight>(this),
			ObjectContainerWithBVH<Mesh>(this),
			ObjectContainer<Group>(this))
		, m_material(
			this,
			ConStruct<Material>(
				"world_material",
				Graphics::Color(0xFF, 0xFF, 0xFF, 0x00),
				0.0f, 0.0f, 0.0f, 1.0f, 0.0f))
		, m_default_material(
			this,
			ConStruct<Material>(
				"world_default_material",
				Graphics::Color::Palette::LightGrey))
		, mp_loader(new Loader(*this))
		, mp_saver(new Saver(*this))
	{}

	Material& World::material()
	{
		return m_material;
	}
	const Material& World::material() const
	{
		return m_material;
	}
	Material& World::defaultMaterial()
	{
		return m_default_material;
	}
	const Material& World::defaultMaterial() const
	{
		return m_default_material;
	}

	Loader& World::loader()
	{
		return *mp_loader;
	}
	const Loader& World::loader() const
	{
		return *mp_loader;
	}
	Saver& World::saver()
	{
		return *mp_saver;
	}
	const Saver& World::saver() const
	{
		return *mp_saver;
	}

	void World::destroyAll()
	{
		container<ObjectType::Texture>().destroyAll();
		container<ObjectType::NormalMap>().destroyAll();
		container<ObjectType::MetalnessMap>().destroyAll();
		container<ObjectType::RoughnessMap>().destroyAll();
		container<ObjectType::EmissionMap>().destroyAll();

		container<ObjectType::Material>().destroyAll();
		container<ObjectType::MeshStructure>().destroyAll();

		container<ObjectType::Camera>().destroyAll();

		container<ObjectType::SpotLight>().destroyAll();
		container<ObjectType::DirectLight>().destroyAll();

		container<ObjectType::Mesh>().destroyAll();
		container<ObjectType::Group>().destroyAll();
	}

	void World::update()
	{
		if (!stateRegister().RequiresUpdate()) return;


		container<ObjectType::Texture>().update();
		container<ObjectType::NormalMap>().update();
		container<ObjectType::MetalnessMap>().update();
		container<ObjectType::RoughnessMap>().update();
		container<ObjectType::EmissionMap>().update();

		container<ObjectType::Material>().update();
		container<ObjectType::MeshStructure>().update();

		container<ObjectType::Camera>().update();

		container<ObjectType::SpotLight>().update();
		container<ObjectType::DirectLight>().update();

		container<ObjectType::Mesh>().update();
		container<ObjectType::Group>().update();

		stateRegister().update();
	}


	template<>
	Handle<MeshStructure> World::generateMesh<World::CommonMesh::Cube>(
		[[maybe_unused]] const CommonMeshParameters<CommonMesh::Cube>& properties)
	{
		auto mesh = container<ObjectType::MeshStructure>().create(
			ConStruct<MeshStructure>("default cube", 8, 4, 0, 12));

		// vertices
		mesh->createVertex(Math::vec3f32(-0.5f, +0.5f, -0.5f));
		mesh->createVertex(Math::vec3f32(-0.5f, +0.5f, +0.5f));
		mesh->createVertex(Math::vec3f32(+0.5f, +0.5f, +0.5f));
		mesh->createVertex(Math::vec3f32(+0.5f, +0.5f, -0.5f));
		mesh->createVertex(Math::vec3f32(-0.5f, -0.5f, -0.5f));
		mesh->createVertex(Math::vec3f32(-0.5f, -0.5f, +0.5f));
		mesh->createVertex(Math::vec3f32(+0.5f, -0.5f, +0.5f));
		mesh->createVertex(Math::vec3f32(+0.5f, -0.5f, -0.5f));

		// texcrds
		mesh->createTexcrd(Math::vec2f32(0.0f, 0.0f));
		mesh->createTexcrd(Math::vec2f32(0.0f, 1.0f));
		mesh->createTexcrd(Math::vec2f32(1.0f, 1.0f));
		mesh->createTexcrd(Math::vec2f32(1.0f, 0.0f));

		// triangles
		mesh->createTriangle({ 1, 2, 0 }, { 1, 2, 0 });
		mesh->createTriangle({ 3, 0, 2 }, { 3, 0, 2 });
		mesh->createTriangle({ 4, 7, 5 }, { 1, 2, 0 });
		mesh->createTriangle({ 6, 5, 7 }, { 3, 0, 2 });
		mesh->createTriangle({ 0, 3, 4 }, { 1, 2, 0 });
		mesh->createTriangle({ 7, 4, 3 }, { 3, 0, 2 });
		mesh->createTriangle({ 2, 1, 6 }, { 1, 2, 0 });
		mesh->createTriangle({ 5, 6, 1 }, { 3, 0, 2 });
		mesh->createTriangle({ 3, 2, 7 }, { 1, 2, 0 });
		mesh->createTriangle({ 6, 7, 2 }, { 3, 0, 2 });
		mesh->createTriangle({ 1, 0, 5 }, { 1, 2, 0 });
		mesh->createTriangle({ 4, 5, 0 }, { 3, 0, 2 });

		return mesh;
	}
	template<>
	Handle<MeshStructure> World::generateMesh<World::CommonMesh::Plane>(
		const CommonMeshParameters<CommonMesh::Plane>& properties)
	{
		RZAssert(properties.sides >= 3, "shape should have at least 3 sides");

		auto mesh = container<ObjectType::MeshStructure>().create(
			ConStruct<MeshStructure>(
				"generated plane",
				properties.sides, properties.sides, 0, properties.sides - 2));

		// vertices
		const float delta_theta = Math::constants<float>::pi * 2.0f / properties.sides;
		const float offset_theta = delta_theta * 0.5f;
		for (uint32_t i = 0; i < properties.sides; i++)
		{
			const auto angle = delta_theta * i + offset_theta;
			mesh->createVertex(
				Math::vec3f32(std::cosf(angle), 0.0f, std::sinf(angle)) *
				Math::vec3f32(properties.width, 0.0f, properties.height));
		}

		// triangles
		for (uint32_t i = 0; i < properties.sides - 2; i++)
		{
			mesh->createTriangle({ 0, i + 2, i + 1 });
		}

		return mesh;
	}
	template<>
	Handle<MeshStructure> World::generateMesh<World::CommonMesh::Sphere>(
		const CommonMeshParameters<CommonMesh::Sphere>& properties)
	{
		auto mesh = container<ObjectType::MeshStructure>().create(
			ConStruct<MeshStructure>("generated sphere"));

		switch (properties.type)
		{
			case CommonMeshParameters<CommonMesh::Sphere>::Type::UVSphere:
			{
				RZAssert(properties.resolution >= 4, "sphere should have at least 4 subdivisions");

				// vertices + normals
				const float d_theta = Math::constants<float>::pi / (properties.resolution / 2);
				const float d_phi = 2.0f * Math::constants<float>::pi / properties.resolution;
				for (uint32_t theta = 0; theta < properties.resolution / 2 - 1; theta++)
				{
					for (uint32_t phi = 0; phi < properties.resolution; phi++)
					{
						Vertex v(0.0f, 1.0f, 0.0f);
						const float a_theta = d_theta * (theta + 1);
						const float a_phi = d_phi * phi;
						v.RotateX(a_theta);
						v.RotateY(a_phi);
						mesh->createVertex(v);

						if (properties.normals)
						{
							mesh->createNormal(v);
						}
					}
				}
				const auto top_v_idx = mesh->createVertex(Math::vec3f32(0.0f, 1.0f, 0.0f));
				const auto bottom_v_idx = mesh->createVertex(Math::vec3f32(0.0f, -1.0f, 0.0f));
				if (properties.normals)
				{
					mesh->createNormal(Math::vec3f32(0.0f, +1.0f, 0.0f));
					mesh->createNormal(Math::vec3f32(0.0f, -1.0f, 0.0f));
				}

				// texture coordinates
				uint32_t top_t_idx = 0, bottom_t_idx = 0;
				if (properties.texture_coordinates)
				{
					for (uint32_t theta = 0; theta < properties.resolution / 2 - 1; theta++)
					{
						for (uint32_t phi = 0; phi < properties.resolution; phi++)
						{
							const float a_theta = d_theta * (theta + 1);
							const float a_phi = d_phi * phi;
							mesh->createTexcrd(Math::vec2f32(
								a_phi * 0.5f * Math::constants<float>::r_pi,
								1.0f - a_theta * Math::constants<float>::r_pi));
						}
						mesh->createTexcrd(Math::vec2f32(
							1.0f,
							1.0f - (d_theta * (theta + 1)) * Math::constants<float>::r_pi));
					}

					for (uint32_t i = 0; i < properties.resolution; i++)
					{
						auto top_idx = mesh->createTexcrd(Math::vec2f32(
							i / float(properties.resolution) + (0.5f / properties.resolution),
							1.0f));
						if (i == 0) top_t_idx = top_idx;
					}
					for (uint32_t i = 0; i < properties.resolution; i++)
					{
						auto bottom_idx = mesh->createTexcrd(Math::vec2f32(
							i / float(properties.resolution) + (0.5f / properties.resolution),
							0.0f));
						if (i == 0) bottom_t_idx = bottom_idx;
					}
				}

				// triangles
				using triple_index_t = MeshStructure::triple_index_t;
				// top and bottom fan
				triple_index_t vn_ids_value{}, t_ids_value{};
				for (uint32_t i = 0; i < properties.resolution; i++)
				{
					const triple_index_t& top_v_ids = vn_ids_value = {
						top_v_idx,
						(i + 1) % properties.resolution,
						i };
					const triple_index_t& top_t_ids = properties.texture_coordinates ? t_ids_value = {
						top_t_idx + i,
						i + 1,
						i } : MeshStructure::ids_unused;
					const triple_index_t& top_n_ids = properties.normals ? top_v_ids : MeshStructure::ids_unused;
					mesh->createTriangle(top_v_ids, top_t_ids, top_n_ids);

					const triple_index_t& bottom_v_ids = vn_ids_value = {
						bottom_v_idx,
						top_v_idx - properties.resolution + i,
						top_v_idx - properties.resolution + (i + 1) % properties.resolution };
					const triple_index_t& bottom_t_ids = properties.texture_coordinates ? t_ids_value = {
						bottom_t_idx + i,
						top_t_idx - properties.resolution + i - 1,
						top_t_idx - properties.resolution + i } : MeshStructure::ids_unused;
					const triple_index_t& bottom_n_ids = properties.normals ? bottom_v_ids : MeshStructure::ids_unused;
					mesh->createTriangle(bottom_v_ids, bottom_t_ids, bottom_n_ids);
				}
				// middle layers
				for (uint32_t theta = 0; theta < properties.resolution / 2 - 2; theta++)
				{
					for (uint32_t phi = 0; phi < properties.resolution; phi++)
					{
						const triple_index_t& v_ids1 = vn_ids_value = {
							theta * properties.resolution + phi,
							theta * properties.resolution + (phi + 1) % properties.resolution,
							(theta + 1) * properties.resolution + (phi + 1) % properties.resolution };
						const triple_index_t& t_ids1 = properties.texture_coordinates ? t_ids_value = {
							theta * (properties.resolution + 1) + phi,
							theta * (properties.resolution + 1) + (phi + 1),
							(theta + 1) * (properties.resolution + 1) + (phi + 1) } : MeshStructure::ids_unused;
						const triple_index_t& n_ids1 = properties.normals ? v_ids1 : MeshStructure::ids_unused;
						mesh->createTriangle(v_ids1, t_ids1, n_ids1);

						const triple_index_t& v_ids2 = vn_ids_value = {
							theta * properties.resolution + phi,
							(theta + 1) * properties.resolution + (phi + 1) % properties.resolution,
							(theta + 1) * properties.resolution + phi };
						const triple_index_t& t_ids2 = properties.texture_coordinates ? t_ids_value = {
							theta * (properties.resolution + 1) + phi,
							(theta + 1) * (properties.resolution + 1) + (phi + 1),
							(theta + 1) * (properties.resolution + 1) + phi } : MeshStructure::ids_unused;
						const triple_index_t& n_ids2 = properties.normals ? v_ids2 : MeshStructure::ids_unused;
						mesh->createTriangle(v_ids2, t_ids2, n_ids2);
					}
				}
				break;
			}
			default:
				RZThrow("failed to generate sphere with unsupported tesselation method");
		}

		return mesh;
	}
	template<>
	Handle<MeshStructure> World::generateMesh<World::CommonMesh::Cone>(
		const CommonMeshParameters<CommonMesh::Cone>& properties)
	{
		RZAssert(properties.side_faces >= 3, "cone should have at least 3 side faces");

		auto mesh = container<ObjectType::MeshStructure>().create(
			ConStruct<MeshStructure>("generated cone"));

		// vertices
		const float delta_phi = Math::constants<float>::pi * 2.0f / properties.side_faces;
		const float offset_phi = delta_phi * 0.5f;
		for (uint32_t i = 0; i < properties.side_faces; i++)
		{
			const auto angle = delta_phi * i + offset_phi;
			mesh->createVertex(Math::vec3f32(std::cosf(angle), 0.0f, std::sinf(angle)));
		}
		const auto apex_v_idx = mesh->createVertex(Math::vec3f32(0.0f, 1.0f, 0.0f)); // apex

		// normals
		for (uint32_t i = 0; i < properties.side_faces; i++)
		{
			const auto angle = delta_phi * i + offset_phi;
			mesh->createNormal(
				Math::vec3f32(0.0f, 1.0f, 0.0f)
				.RotatedX(0.25f * Math::constants<float>::pi)
				.RotatedY(angle + 0.5f * Math::constants<float>::pi));
			mesh->createNormal(
				Math::vec3f32(0.0f, 1.0f, 0.0f).
				RotatedX(0.25f * Math::constants<float>::pi).
				RotatedY(angle + 0.5f * Math::constants<float>::pi + 0.5f * delta_phi));
		}

		// triangles
		// side faces
		using triple_index_t = MeshStructure::triple_index_t;
		triple_index_t v_ids_value{}, n_ids_value{};
		for (uint32_t i = 0; i < properties.side_faces; i++)
		{
			const triple_index_t& v_ids = v_ids_value = { apex_v_idx, (i + 1) % properties.side_faces, i };
			const triple_index_t& n_ids = properties.normals ? n_ids_value = {
				(i * 2 + 1) % (properties.side_faces * 2),
				((i + 1) * 2) % (properties.side_faces * 2),
				i * 2 } : MeshStructure::ids_unused;
			mesh->createTriangle(v_ids, MeshStructure::ids_unused, n_ids);
		}
		// base
		for (uint32_t i = 0; i < properties.side_faces - 2; i++)
		{
			mesh->createTriangle({
				0,
				i + 1,
				(i + 2) % properties.side_faces });
		}

		return mesh;
	}
	template<>
	Handle<MeshStructure> World::generateMesh<World::CommonMesh::Cylinder>(
		const CommonMeshParameters<CommonMesh::Cylinder>& properties)
	{
		RZAssert(properties.faces >= 3, "cylinder should have at least 3 faces");

		const auto vertices_num = properties.faces * 2;
		const auto tris_num = (properties.faces - 2) * 2 + properties.faces * 2;
		auto mesh = container<ObjectType::MeshStructure>().create(
			ConStruct<MeshStructure>(
				"generated cylinder",
				vertices_num, 1, vertices_num * 2, tris_num));

		mesh->createTexcrd(Math::vec2f32(0.5f, 0.5f));

		// vertices + normals
		const float delta_theta = Math::constants<float>::pi * 2.0f / properties.faces;
		const float offset_theta = delta_theta * 0.5f;
		for (uint32_t i = 0; i < properties.faces; i++)
		{
			const auto angle = delta_theta * i + offset_theta;
			mesh->createVertex(Math::vec3f32(std::cosf(angle), -1.0f, std::sinf(angle)));
			mesh->createVertex(Math::vec3f32(std::cosf(angle), +1.0f, std::sinf(angle)));

			if (properties.normals)
				mesh->createNormal(Math::vec3f32(1.0f, 0.0f, 0.0f).RotatedY(angle));
		}

		auto vertex_idx = [&vertices_num](const uint32_t idx)
		{
			return idx % vertices_num;
		};

		// triangles
		for (uint32_t i = 0; i < properties.faces - 2; i++)
		{
			// bottom
			mesh->createTriangle({
				0,
				vertex_idx((i + 1) * 2),
				vertex_idx((i + 2) * 2) });
			// top
			mesh->createTriangle({
				1,
				vertex_idx((i + 2) * 2 + 1),
				vertex_idx((i + 1) * 2 + 1) });
		}
		if (properties.normals)
		{
			for (uint32_t i = 0; i < properties.faces; i++)
			{
				// side
				mesh->createTriangle({
					vertex_idx(i * 2),
					vertex_idx(i * 2 + 1),
					vertex_idx((i + 1) * 2 + 1) },
					{ 0, 0, 0 }, { i, i, (i + 1) % properties.faces });
				mesh->createTriangle({
					vertex_idx(i * 2),
					vertex_idx((i + 1) * 2 + 1),
					vertex_idx((i + 1) * 2) },
					{ 0, 0, 0 }, { i, (i + 1) % properties.faces, (i + 1) % properties.faces });
			}
		}
		else
		{
			for (uint32_t i = 0; i < properties.faces; i++)
			{
				// side
				mesh->createTriangle({
					vertex_idx(i * 2),
					vertex_idx(i * 2 + 1),
					vertex_idx((i + 1) * 2 + 1) });
				mesh->createTriangle({
					vertex_idx(i * 2),
					vertex_idx((i + 1) * 2 + 1),
					vertex_idx((i + 1) * 2) });
			}
		}

		return mesh;
	}
	template<>
	Handle<MeshStructure> World::generateMesh<World::CommonMesh::Torus>(
		const CommonMeshParameters<CommonMesh::Torus>& properties)
	{
		RZAssert(
			properties.minor_resolution >= 3 && properties.major_resolution >= 3,
			"resolution should be at least 3");

		auto mesh = container<ObjectType::MeshStructure>().create(
			ConStruct<MeshStructure>("generated torus"));

		// vertices + normals
		const float d_phi = Math::constants<float>::pi * 2.0f / properties.major_resolution;
		const float offset_phi = d_phi * 0.5f;
		const float d_theta = Math::constants<float>::pi * 2.0f / properties.minor_resolution;
		for (uint32_t M = 0; M < properties.major_resolution; M++)
		{
			const auto a_phi = d_phi * M + offset_phi;
			for (uint32_t m = 0; m < properties.minor_resolution; m++)
			{
				const auto a_theta = d_theta * m;
				auto major_center = Math::vec3f32(1.0f, 0.0f, 0.0f).RotatedY(a_phi);
				auto normal = Math::vec3f32(1.0f, 0.0f, 0.0f).RotatedZ(-a_theta).RotatedY(a_phi);
				mesh->createVertex(
					major_center * properties.major_radious +
					normal * properties.minor_radious);

				if (properties.normals)
					mesh->createNormal(normal);
			}
		}
		// texcrds
		if (properties.texture_coordinates)
		{
			for (uint32_t M = 0; M <= properties.major_resolution; M++)
			{
				for (uint32_t m = 0; m <= properties.minor_resolution; m++)
				{
					mesh->createTexcrd(Math::vec2f32(
						M / float(properties.major_resolution),
						m / float(properties.minor_resolution)));
				}
			}
		}

		// triangles
		using triple_index_t = MeshStructure::triple_index_t;
		triple_index_t vn_ids_value{}, t_ids_value{};
		for (uint32_t M = 0; M < properties.major_resolution; M++)
		{
			for (uint32_t m = 0; m < properties.minor_resolution; m++)
			{
				const auto& v_ids1 = vn_ids_value = {
					M * properties.minor_resolution + m,
					M * properties.minor_resolution + (m + 1) % properties.minor_resolution,
					((M + 1) % properties.major_resolution) * properties.minor_resolution +
						(m + 1) % properties.minor_resolution };
				const auto& t_ids1 = t_ids_value = properties.texture_coordinates ? triple_index_t{
					M * (properties.minor_resolution + 1) + m,
					M * (properties.minor_resolution + 1) + m + 1,
					(M + 1) * (properties.minor_resolution + 1) + m + 1 } : MeshStructure::ids_unused;
				const auto& n_ids1 = properties.normals ? v_ids1 : MeshStructure::ids_unused;
				mesh->createTriangle(v_ids1, t_ids1, n_ids1);

				const auto& v_ids2 = vn_ids_value = {
					M * properties.minor_resolution + m,
					((M + 1) % properties.major_resolution) * properties.minor_resolution +
						(m + 1) % properties.minor_resolution,
					((M + 1) % properties.major_resolution) * properties.minor_resolution + m };
				const auto& t_ids2 = t_ids_value = properties.texture_coordinates ? triple_index_t{
					M * (properties.minor_resolution + 1) + m,
					(M + 1) * (properties.minor_resolution + 1) + m + 1,
					(M + 1) * (properties.minor_resolution + 1) + m } : MeshStructure::ids_unused;
				const auto& n_ids2 = properties.normals ? v_ids2 : MeshStructure::ids_unused;
				mesh->createTriangle(v_ids2, t_ids2, n_ids2);
			}
		}

		return mesh;
	}
}
