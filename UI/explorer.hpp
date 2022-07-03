#pragma once

#include "rayzath.h"

#include "scene.hpp"
#include "properties.hpp"
#include "viewport.hpp"

#include <unordered_set>

namespace RZ = RayZath::Engine;

namespace RayZath::UI::Windows
{
	class Explorer
	{
	private:
		Scene& mr_scene;
		Properties m_properties;

		RZ::Handle<RZ::Mesh> m_current_object;
		RZ::Handle<RZ::Group> m_current_group;
		std::unordered_map<uint32_t, bool> m_object_ids, m_group_ids; // alraedy drawn objects
		std::reference_wrapper<Viewports> m_viewports;

	public:
		Explorer(Scene& scene, Viewports& viewports);

		void update();

	private:
		void listCameras();

		void listLights();

		void listObject(const RZ::Handle<RZ::Mesh>& object);
		void objectTree(const RZ::Handle<RZ::Group>& group);
		void listObjects();

		void listMaterials();

		void listTextures();
		void listNormalMaps();
		void listMetalnessMaps();
		void listRoughnessMaps();
		void listEmissionMaps();
		void listMaps();
	};
}