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

		RZ::Camera* mp_camera;


	public:
		Scene(Application& app);
		~Scene();


	public:
		void Render();
		const Graphics::Bitmap& GetRender();
		void ResizeRender(size_t width, size_t height);

	private:
		void CreateCube(RZ::World* world, const RZ::ConStruct<RZ::Mesh>& conStruct);
		void CreateRoom(RZ::World* world);
	};
}

#endif // !SCENE_H