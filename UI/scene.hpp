#pragma once

#include "rayzath.h"

namespace RZ = RayZath::Engine;

namespace RayZath::UI
{
	enum class CommonMesh
	{
		Plane,
		Sphere,
		Cone,
		Cylinder,
		Torus
	};
	template <CommonMesh T>
	struct CommonMeshParameters;
	template<>
	struct CommonMeshParameters<CommonMesh::Plane>
	{
		uint32_t sides = 4;
		float width = 1.0f, height = 1.0f;

	public:
		CommonMeshParameters(const uint32_t sides = 4, const float width = 1.0f, const float height = 1.0f)
			: sides(sides)
			, width(width)
			, height(height)
		{}
	};
	template<>
	struct CommonMeshParameters<CommonMesh::Sphere>
	{
		uint32_t resolution = 16;
		bool normals = true;
		bool texture_coordinates = true;
		enum class Type
		{
			UVSphere,
			Icosphere
		} type = Type::UVSphere;
	};
	template<>
	struct CommonMeshParameters<CommonMesh::Cone>
	{
		uint32_t side_faces = 16;
		bool normals = true;
		bool texture_coordinates = true;
	};
	template<>
	struct CommonMeshParameters<CommonMesh::Cylinder>
	{
		uint32_t faces = 16;
		bool normals = true;

	public:
		CommonMeshParameters(const uint32_t faces = 16, const bool normals = true)
			: faces(faces)
			, normals(normals)
		{}
	};
	template<>
	struct CommonMeshParameters<CommonMesh::Torus>
	{
		uint32_t minor_resolution = 16, major_resolution = 32;
		float minor_radious = 0.25f, major_radious = 1.0f;
		bool normals = true;
		bool texture_coordinates = true;
	};

	class Scene
	{
	private:
	public:
		RZ::Engine& mr_engine;
		RZ::World& mr_world;

		Scene();

		void init();
		void createDefaultScene();

		void render();
		void update(const float elapsed_time);

		template <CommonMesh T>
		RZ::Handle<RZ::MeshStructure> Create(const CommonMeshParameters<T>& parameters);
	};
}