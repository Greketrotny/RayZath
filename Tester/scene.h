#ifndef SCENE_H
#define SCENE_H

#include "rayzath.h"
namespace RZ = RayZath;

#include "bitmap.h"

namespace Tester
{
	class Application;

	class Scene
	{
	private:
		Application& mr_app;

	public:
		RZ::Engine& mr_engine;
		RZ::World& mr_world;

		RZ::Handle<RZ::Camera> m_camera;

		RZ::Handle<RZ::Mesh> cube;
		RZ::Handle<RZ::Mesh> teapot, bunny;

		std::vector<std::string> m_scene_files;
		std::string m_base_scene_path;

	public:
		Scene(Application& app);


	public:
		void Init();
		void Render();
		const Graphics::Bitmap& GetRender();
		void ResizeRender(uint32_t width, uint32_t height);
		void Update(const float elapsed_time);

		void LoadScene(size_t scene_id = 0u);

	private:
		RZ::Handle<RZ::Mesh> CreateCube(
			RZ::World& world, RZ::ConStruct<RZ::Mesh> conStruct);

		/*void CreateTessellatedSphere(
			RZ::World* world,
			const RZ::ConStruct<RZ::Mesh>& conStruct,
			const uint32_t& resolution = 8u);*/
	};
}

#endif // !SCENE_H