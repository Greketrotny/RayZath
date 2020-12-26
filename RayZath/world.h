#ifndef WORLD_H
#define WORLD_H

#include "object_container.h"
#include "bvh.h"

#include "camera.h"

#include "point_light.h"
#include "spot_light.h"
#include "direct_light.h"

#include "mesh.h"
#include "sphere.h"

namespace RayZath
{
	class World : public Updatable
	{
	private:
		ObjectContainer<Texture> m_textures;
		ObjectContainer<Material> m_materials;
		ObjectContainer<MeshStructure> m_mesh_structures;

		ObjectContainer<Camera> m_cameras;

		ObjectContainer<PointLight> m_point_lights;
		ObjectContainer<SpotLight> m_spot_lights;
		ObjectContainer<DirectLight> m_direct_lights;

		ObjectContainerWithBVH<Mesh> m_meshes;
		ObjectContainerWithBVH<Sphere> m_spheres;

		Material m_material;
		Material m_default_material;

		/*class Cache
		{
			 // An "envelope" type which up-casts to the right ObjectTable<T> 
			 // if we have a type parameter T. 
			 struct ObjectTables : ObjectTable<ObjTypeA>,  
								   ObjectTable<ObjTypeB>, 
								   ObjectTable<ObjTypeC> {};

			 ObjectTables tables; 
			public:

			template <typename T>
			void getObjectWithId(T &objectBuffer, int id)
			{ 
				// C++ does the work here
				ObjectTable<T> &o=tables;
				t.getObjectWithId(objectBuffer, id);
			}
		};*/


	private:
		World(
			const uint32_t& maxCamerasCount = 1u, 
			const uint32_t& maxLightsCount = 16u, 
			const uint32_t& maxRenderObjectsCount = 0x1000u);
		~World();


	public:
		ObjectContainer<Texture>& GetTextures();
		const ObjectContainer<Texture>& GetTextures() const;
		ObjectContainer<Material>& GetMaterials();
		const ObjectContainer<Material>& GetMaterials() const;
		ObjectContainer<MeshStructure>& GetMeshStructures();
		const ObjectContainer<MeshStructure>& GetMeshStructures() const;

		ObjectContainer<Camera>& GetCameras();
		const ObjectContainer<Camera>& GetCameras() const;

		ObjectContainer<PointLight>& GetPointLights();
		const ObjectContainer<PointLight>& GetPointLights() const;
		ObjectContainer<SpotLight>& GetSpotLights();
		const ObjectContainer<SpotLight>& GetSpotLights() const;
		ObjectContainer<DirectLight>& GetDirectLights();
		const ObjectContainer<DirectLight>& GetDirectLights() const;

		ObjectContainerWithBVH<Mesh>& GetMeshes();
		const ObjectContainerWithBVH<Mesh>& GetMeshes() const;
		ObjectContainerWithBVH<Sphere>& GetSpheres();
		const ObjectContainerWithBVH<Sphere>& GetSpheres() const;

		Material& GetMaterial();
		const Material& GetMaterial() const;
		Material& GetDefaultMaterial();
		const Material& GetDefaultMaterial() const;

		void DestroyAllComponents();

		void Update() override;


		friend class Engine;
	};
}

#endif // !WORLD_H