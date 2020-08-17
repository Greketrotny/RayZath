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
		RZ::Engine& mr_engine;
		RZ::World& mr_world;

		RZ::Camera* mp_camera;


	public:
		Scene(Application& app);
		~Scene();


	public:
		void Render();
		//const Graphics::Bitmap& GetRender();

	private:
		//void CreateCube(RZ::World* world, const RZ::ConStruct<Mesh>& conStruct);
		//void CreateRoom(RZ::World* world);
	};
}

#endif // !SCENE_H