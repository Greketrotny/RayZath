module;

#include "rayzath.h"

#include <memory>
#include <unordered_set>

export module rz.ui.windows.explorer;

import rz.ui.scene;
import rz.ui.windows.properties;

namespace RZ = RayZath::Engine;

namespace RayZath::UI::Windows
{
	export class Explorer
	{
	private:
		Scene& mr_scene;
		Properties m_properties;

		std::unordered_map<uint32_t, bool> m_object_ids, m_group_ids; // alraedy drawn objects

	public:
		Explorer(Scene& scene);

		void update();

	private:
		void listCameras();

		void listLights();

		void listObject(const RZ::Handle<RZ::Mesh>& object);
		void objectTree(const RZ::Handle<RZ::Group>& group);
		void listObjects();

		void listMaterials();
	};
}