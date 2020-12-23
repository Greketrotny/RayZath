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


	public:
		Scene(Application& app);
		~Scene();


	public:
		void Render();
		const Graphics::Bitmap& GetRender();
		void ResizeRender(uint32_t width, uint32_t height);

	private:
		RZ::Handle<RZ::Mesh> CreateCube(
			RZ::World& world, RZ::ConStruct<RZ::Mesh> conStruct);

		void CreateRoom(
			RZ::World& world,
			const RZ::ConStruct<RZ::RenderObject>& conStruct);

		RZ::Handle<RZ::Mesh> CreateLightPlane(
			RZ::World& world,
			RZ::ConStruct<RZ::Mesh> con_struct,
			const Graphics::Color& color);
		/*void CreateTessellatedSphere(
			RZ::World* world,
			const RZ::ConStruct<RZ::Mesh>& conStruct,
			const uint32_t& resolution = 8u);

		RZ::Mesh* CreateRoundedCube(
			RZ::World& world,
			const RZ::ConStruct<RZ::Mesh>& con_struct);*/

	};
}

#endif // !SCENE_H