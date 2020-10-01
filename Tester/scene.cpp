#include "scene.h"
#include "application.h"

Graphics::Bitmap GenerateColorBitmap()
{
	unsigned int resolution = 32;
	Graphics::Bitmap bitmap(resolution * 8, resolution);

	unsigned char hs = 0xFF;
	unsigned char ls = 0x22;
	//std::vector<Graphics::Color> colors{
	//	Graphics::Color(hs, ls, ls),	// red
	//	Graphics::Color(ls, hs, ls),	// green
	//	Graphics::Color(hs, ls, hs),	// magenta
	//	Graphics::Color(hs, hs, ls),	// yellow
	//	Graphics::Color(ls, ls, hs),	// blue
	//	Graphics::Color(ls, hs, hs),	// cyan
	//	Graphics::Color(hs, hs, hs),	// white
	//	Graphics::Color(ls, ls, ls),	// dark grey
	//};
	std::vector<Graphics::Color> colors{
		//Graphics::Color(hs, ls, ls),	// red
		Graphics::Color(hs, hs, hs),	// white
		Graphics::Color(ls, hs, ls),	// green
		Graphics::Color(hs, hs, hs),	// white
		Graphics::Color(hs, hs, hs),	// white
		Graphics::Color(hs, hs, hs),	// white
		Graphics::Color(hs, hs, hs),	// white
		Graphics::Color(hs, hs, hs),	// white
		Graphics::Color(ls, ls, ls),	// dark grey
	};

	for (int i = 0; i < 8; ++i)
	{
		for (unsigned int x = 0; x < resolution; ++x)
		{
			for (unsigned int y = 0; y < resolution; ++y)
			{
				if ((x % 2 == 0) ^ (y % 2 == 0)) bitmap.SetPixel(resolution * i + x, y, colors[i]);
				else bitmap.SetPixel(resolution * i + x, y, Graphics::Color::BlendAverage(colors[i], Graphics::Color(0x00, 0x00, 0x00)));

				if (x == 2 && y == 2)
					bitmap.SetPixel(x, y, Graphics::Color(0x00, 0x00, 0xFF));
			}
		}
	}
	
	return bitmap;
}
Graphics::Bitmap GenerateBitmap()
{
	unsigned int resolution = 8u;
	Graphics::Bitmap bitmap(resolution, resolution);

	for (unsigned int x = 0; x < resolution; ++x)
	{
		for (unsigned int y = 0; y < resolution; ++y)
		{
			// white
			if ((x % 2 == 0) ^ (y % 2 == 0)) bitmap.SetPixel(x, y, Graphics::Color(0xFF, 0xFF, 0xFF, 0xFF));
			else bitmap.SetPixel(x, y, Graphics::Color(0x20, 0xFF, 0x20, 0x10));
			//if ((x % 2 == 0) ^ (y % 2 == 0)) bitmap.SetPixel(x, y, Graphics::Color(0xFF, 0x88, 0x88));
			//else bitmap.SetPixel(x, y, Graphics::Color(0xFF, 0x00, 0x00));
		}
	}
	return bitmap;

	//unsigned int resolution = 8;
	//Graphics::Bitmap bitmap(3 * resolution, 2 * resolution);

	//for (unsigned int x = 0; x < resolution; ++x)
	//{
	//	for (unsigned int y = 0; y < resolution; ++y)
	//	{
	//		// white
	//		if ((x % 2 == 0) ^ (y % 2 == 0)) bitmap.SetPixel(x, y, Graphics::Color(0xFF, 0xFF, 0xFF));
	//		else bitmap.SetPixel(x, y, Graphics::Color(0x22, 0x22, 0x22));

	//		// red
	//		if (((x + resolution) % 2 == 0) ^ (y % 2 == 0)) bitmap.SetPixel(x, y, Graphics::Color(0xFF, 0x22, 0x22));
	//		else bitmap.SetPixel(x + resolution, y, Graphics::Color(0x22, 0x22, 0x22));

	//		// green
	//		if (((x + 2 * resolution) % 2 == 0) ^ (y % 2 == 0)) bitmap.SetPixel(x, y, Graphics::Color(0x22, 0xFF, 0x22));
	//		else bitmap.SetPixel(x + 2 * resolution, y, Graphics::Color(0x22, 0x22, 0x22));

	//		// blue
	//		if ((x % 2 == 0) ^ ((y + resolution) % 2 == 0)) bitmap.SetPixel(x, y, Graphics::Color(0x22, 0x22, 0xFF));
	//		else bitmap.SetPixel(x, y + resolution, Graphics::Color(0x22, 0x22, 0x22));

	//		// yellow
	//		if (((x + resolution) % 2 == 0) ^ ((y + resolution) % 2 == 0)) bitmap.SetPixel(x, y, Graphics::Color(0xFF, 0xFF, 0x22));
	//		else bitmap.SetPixel(x+ resolution, y + resolution, Graphics::Color(0x22, 0x22, 0x22));

	//		// ?
	//		if (((x + 2 * resolution) % 2 == 0) ^ ((y + resolution) % 2 == 0)) bitmap.SetPixel(x, y, Graphics::Color(0x22, 0xFF, 0xFF));
	//		else bitmap.SetPixel(x + 2 * resolution, y + resolution, Graphics::Color(0x22, 0x22, 0x22));
	//	}
	//}
	//return bitmap;
}


namespace Tester
{
	Scene::Scene(Application& app)
		: mr_app(app)
		, mr_engine(RZ::Engine::GetInstance())
		, mr_world(mr_engine.GetWorld())
	{
		//// cameras
		//mp_camera = mr_world.GetCameras().CreateObject(RZ::ConStruct<RZ::Camera>(
		//	RZ::ConStruct<RZ::WorldObject>(L"camera 1"),
		//	Math::vec3<float>(0.0f, 5.0f, -12.0f),
		//	Math::vec3<float>(-0.3f, 0.0f, 0.0f),
		//	1200, 700,
		//	Math::angle<Math::deg, float>(100.0f),
		//	10.0f, 0.001f, true));

		////// point lights
		////mr_world.GetPointLights().CreateObject(RZ::ConStruct<RZ::PointLight>(
		////	RZ::ConStruct<RZ::WorldObject>(L"point light 1"),
		////	Math::vec3<float>(0.0f, 3.0f, 0.0f),
		////	Graphics::Color(0xFF, 0xFF, 0xFF),
		////	0.2f, 100.0f));

		////mp_world->Lights.CreateObject(ConStruct<PointLight>(
		////	ConStruct<WorldObject>(L"point light 2"),
		////	Math::vec3<float>(0.0f, 5.0f, 0.0f),
		////	Graphics::Color(0xFF, 0xFF, 0xFF),
		////	0.2f, 100.0f));

		////// spot lights
		////mr_world.GetSpotLights().CreateObject(RZ::ConStruct<RZ::SpotLight>(
		////	RZ::ConStruct<RZ::WorldObject>(L"spotlight1"),
		////	Math::vec3<float>(0.0f, 4.0f, -4.0f),
		////	Math::vec3<float>(0.0f, -1.0f, 1.0f),
		////	Graphics::Color(0xFF, 0x10, 0x10),
		////	0.25f, 500.0f, 0.6f, 2.0f));

		////// direct lights
		////mr_world.GetDirectLights().CreateObject(RZ::ConStruct<RZ::DirectLight>(
		////	RZ::ConStruct<RZ::WorldObject>(L"direct1"),
		////	Math::vec3<float>(-1.0f, -1.0f, 0.0f),
		////	Graphics::Color(0xFF, 0xFF, 0xFF),
		////	20.0f, 0.05f));


		//// sphere1
		//RZ::Sphere* s1 = mr_world.GetSpheres().CreateObject(RZ::ConStruct<RZ::Sphere>(
		//	RZ::ConStruct<RZ::RenderObject>(
		//		RZ::ConStruct<RZ::WorldObject>(L"Sphere 1"),
		//		Math::vec3<float>(-4.0f, 1.0f, 2.0f),
		//		Math::vec3<float>(0.0f, 0.0f, 0.0f),
		//		Math::vec3<float>(0.0f, 0.0f, 0.0f),
		//		Math::vec3<float>(1.0f, 1.0f, 1.0f),
		//		RZ::Material(0.6f))));
		//s1->SetColor(Graphics::Color(0x10, 0xFF, 0x40, 0x00));

		//// sphere2
		//RZ::Sphere* s2 = mr_world.GetSpheres().CreateObject(RZ::ConStruct<RZ::Sphere>(
		//	RZ::ConStruct<RZ::RenderObject>(
		//		RZ::ConStruct<RZ::WorldObject>(L"Sphere 2"),
		//		Math::vec3<float>(-1.25f, 1.0f, -3.0f),
		//		Math::vec3<float>(0.0f, 0.0f, 0.0f),
		//		Math::vec3<float>(0.0f, 0.0f, 0.0f),
		//		Math::vec3<float>(1.0f, 1.0f, 1.0f),
		//		RZ::Material(0.0f, 0.0f, 1.0f, 1.5f))));
		//s2->SetColor(Graphics::Color(0xFF, 0xA0, 0xA0, 0x00));

		//// light sphere
		//RZ::Sphere* s3 = mr_world.GetSpheres().CreateObject(RZ::ConStruct<RZ::Sphere>(
		//	RZ::ConStruct<RZ::RenderObject>(
		//		RZ::ConStruct<RZ::WorldObject>(L"light sphere"),
		//		Math::vec3<float>(-5.0f, 5.0f, 0.0f),
		//		Math::vec3<float>(0.0f, 0.0f, -0.7f),
		//		Math::vec3<float>(0.0f, 0.0f, 0.0f),
		//		Math::vec3<float>(1.0f, 0.1f, 5.0f),
		//		RZ::Material(0.0f, 0.0f, 0.0f, 0.0f, 50.0f))));
		//s3->SetColor(Graphics::Color(0xFF, 0xFF, 0x40));
		//// light sphere 2
		//RZ::Sphere* s4 = mr_world.GetSpheres().CreateObject(RZ::ConStruct<RZ::Sphere>(
		//	RZ::ConStruct<RZ::RenderObject>(
		//		RZ::ConStruct<RZ::WorldObject>(L"light sphere 2"),
		//		Math::vec3<float>(5.0f, 5.0f, 0.0f),
		//		Math::vec3<float>(0.0f, 0.0f, 0.7f),
		//		Math::vec3<float>(0.0f, 0.0f, 0.0f),
		//		Math::vec3<float>(1.0f, 0.1f, 5.0f),
		//		RZ::Material(0.0f, 0.0f, 0.0f, 0.0f, 50.0f))));
		//s4->SetColor(Graphics::Color(0x40, 0xFF, 0xFF));

		//// create bitmap1
		//Graphics::Bitmap bm(16, 16);
		//for (int x = 0; x < bm.GetWidth(); x++)
		//{
		//	for (int y = 0; y < bm.GetHeight(); y++)
		//	{
		//		if ((x % 2 == 0) ^ (y % 2 == 0))
		//		{
		//			/*bm.SetPixel(x, y,
		//				Graphics::Color(
		//					x / float(bm.GetWidth()) * 255.0f,
		//					y / float(bm.GetHeight()) * 255.0f,
		//					0x04,
		//					0x00));*/
		//			bm.SetPixel(x, y,
		//				Graphics::Color(
		//					0xFF,
		//					0xFF,
		//					0xFF,
		//					0x40));
		//		}
		//		else
		//		{
		//			bm.SetPixel(x, y, Graphics::Color(0x80, 0x80, 0x80));
		//		}
		//	}
		//}
		//s1->LoadTexture(RZ::Texture(bm, RZ::Texture::FilterMode::Point));

		//// create bitmap2
		//Graphics::Bitmap bm2(16, 16);
		//for (int x = 0; x < bm2.GetWidth(); x++)
		//{
		//	for (int y = 0; y < bm2.GetHeight(); y++)
		//	{
		//		if ((x % 2 == 0) ^ (y % 2 == 0))
		//		{
		//			bm2.SetPixel(x, y, Graphics::Color(0xFF, 0xFF, 0xFF));
		//		}
		//		else
		//		{
		//			bm2.SetPixel(x, y, Graphics::Color(0xA0, 0xA0, 0xA0));
		//		}
		//	}
		//}
		////s2->LoadTexture(RZ::Texture(bm2, RZ::Texture::FilterMode::Point));

		//// cube1
		///*CreateCube(&mr_world,
		//	RZ::ConStruct<RZ::Mesh>(
		//		RZ::ConStruct<RZ::RenderObject>(
		//			RZ::ConStruct<RZ::WorldObject>(L"Mesh1"),
		//			Math::vec3<float>(1.2f, 1.73f, 1.0f),
		//			Math::vec3<float>(0.0f, 0.67f, -1.0f),
		//			Math::vec3<float>(0.0f, 0.0f, 0.0f),
		//			Math::vec3<float>(1.0f, 1.0f, 1.0f),
		//			RZ::Material(0.0f, 0.001f, 0.5f, 1.5f)),
		//		8u, 12, 4u));*/
		//CreateCube(&mr_world,
		//	RZ::ConStruct<RZ::Mesh>(
		//		RZ::ConStruct<RZ::RenderObject>(
		//			RZ::ConStruct<RZ::WorldObject>(L"Mesh1"),
		//			Math::vec3<float>(1.2f, 1.73f, 1.0f),
		//			Math::vec3<float>(0.0f, 0.0f, 0.0f),
		//			Math::vec3<float>(0.0f, 0.0f, 0.0f),
		//			Math::vec3<float>(1.0f, 1.0f, 1.0f),
		//			RZ::Material(0.0f, 0.0001f, 0.5f, 1.5f)),
		//		8u, 12, 4u));

		//CreateRoom(&mr_world);

		// cameras
		mp_camera = mr_world.GetCameras().CreateObject(RZ::ConStruct<RZ::Camera>(
			RZ::ConStruct<RZ::WorldObject>(L"camera 1"),
			//Math::vec3<float>(0.0f, 4.0f, -16.0f),
			Math::vec3<float>(0.0f, 2.0f, -4.0f),
			Math::vec3<float>(-0.5f, 0.0f, 0.0f),
			1200, 700,
			Math::angle<Math::deg, float>(100.0f),
			10.0f, 0.001f, true));

		// point lights
		/*mr_world.GetPointLights().CreateObject(RZ::ConStruct<RZ::PointLight>(
			RZ::ConStruct<RZ::WorldObject>(L"point light 1"),
			Math::vec3<float>(3.0f, 3.0f, 0.0f),
			Graphics::Color(0xFF, 0xFF, 0xFF),
			0.2f, 100.0f));*/

		//// light sphere
		//RZ::Sphere* s3 = mr_world.GetSpheres().CreateObject(RZ::ConStruct<RZ::Sphere>(
		//	RZ::ConStruct<RZ::RenderObject>(
		//		RZ::ConStruct<RZ::WorldObject>(L"light sphere"),
		//		Math::vec3<float>(-5.0f, 5.0f, 0.0f),
		//		Math::vec3<float>(0.0f, 0.0f, -0.7f),
		//		Math::vec3<float>(0.0f, 0.0f, 0.0f),
		//		Math::vec3<float>(1.0f, 0.1f, 5.0f),
		//		RZ::Material(0.0f, 0.0f, 0.0f, 0.0f, 50.0f))));
		//s3->SetColor(Graphics::Color(0xFF, 0xFF, 0x40));
		//// light sphere 2
		//RZ::Sphere* s4 = mr_world.GetSpheres().CreateObject(RZ::ConStruct<RZ::Sphere>(
		//	RZ::ConStruct<RZ::RenderObject>(
		//		RZ::ConStruct<RZ::WorldObject>(L"light sphere 2"),
		//		Math::vec3<float>(5.0f, 5.0f, 0.0f),
		//		Math::vec3<float>(0.0f, 0.0f, 0.7f),
		//		Math::vec3<float>(0.0f, 0.0f, 0.0f),
		//		Math::vec3<float>(1.0f, 0.1f, 5.0f),
		//		RZ::Material(0.0f, 0.0f, 0.0f, 0.0f, 50.0f))));
		//s4->SetColor(Graphics::Color(0x40, 0xFF, 0xFF));

		/*RZ::Sphere* s1 = mr_world.GetSpheres().CreateObject(RZ::ConStruct<RZ::Sphere>(
			RZ::ConStruct<RZ::RenderObject>(
				RZ::ConStruct<RZ::WorldObject>(L"sphere 1"),
				Math::vec3<float>(0.0f, 0.0f, -3.0f),
				Math::vec3<float>(0.0f, 0.0f, 0.0f),
				Math::vec3<float>(0.0f, 0.0f, 0.0f),
				Math::vec3<float>(1.0f, 1.0f, 1.0f),
				RZ::Material(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)),
			1.0f, Graphics::Color(0xFF, 0x40, 0x40, 0x00)));*/


		// [>] Create cubes
		const int count = 4u;
		const float space = 8.0f / count;
		const float scale = space / 4.0f;
		for (int x = 0; x < count; x++)
		{
			for (int y = 0; y < count; y++)
			{
				for (int z = 0; z < count; z++)
				{
					/*CreateCube(&mr_world, RZ::ConStruct<RZ::Mesh>(
						RZ::ConStruct<RZ::RenderObject>(
							RZ::ConStruct<RZ::WorldObject>(
								L"cube" + std::to_wstring(x * count * count + y * count + z)),
							Math::vec3<float>(
								(float(x) - float(count - 1) / 2.0f) * space, 
								(float(y) - float(count - 1) / 2.0f) * space, 
								(float(z) - float(count - 1) / 2.0f) * space),
							Math::vec3<float>(0.0f, Math::constants<float>::Pi / 4.0f, 0.0f),
							Math::vec3<float>(0.0f, 0.0f, 0.0f),
							Math::vec3<float>(scale, scale, scale),
							RZ::Material(0.0f))));*/
					/*CreateCube(&mr_world, RZ::ConStruct<RZ::Mesh>(
						RZ::ConStruct<RZ::RenderObject>(
							RZ::ConStruct<RZ::WorldObject>(
								L"cube" + std::to_wstring(x * count * count + y * count + z)),
							Math::vec3<float>(
								(((rand() % RAND_MAX) / float(RAND_MAX)) - 0.5f) * 5.0f,
								(((rand() % RAND_MAX) / float(RAND_MAX)) - 0.5f) * 5.0f,
								(((rand() % RAND_MAX) / float(RAND_MAX)) - 0.5f) * 5.0f),
							Math::vec3<float>(0.0f, Math::constants<float>::Pi / 4.0f, 0.0f),
							Math::vec3<float>(0.0f, 0.0f, 0.0f),
							Math::vec3<float>(scale, scale, scale),
							RZ::Material(0.75f))));*/

					//RZ::Sphere* s1 = mr_world.GetSpheres().CreateObject(RZ::ConStruct<RZ::Sphere>(
					//	RZ::ConStruct<RZ::RenderObject>(
					//		RZ::ConStruct<RZ::WorldObject>(
					//			L"sphere" + std::to_wstring(x * count * count + y * count + z)),
					//		//Math::vec3<float>(x * space, y * space, z * space),
					//		Math::vec3<float>(
					//			(((rand() % RAND_MAX) / float(RAND_MAX)) - 0.5f) * 5.0f,
					//			(((rand() % RAND_MAX) / float(RAND_MAX)) - 0.5f) * 5.0f,
					//			(((rand() % RAND_MAX) / float(RAND_MAX)) - 0.5f) * 5.0f),
					//		Math::vec3<float>(0.0f, 0.0f, 0.0f),
					//		Math::vec3<float>(0.0f, 0.0f, 0.0f),
					//		Math::vec3<float>(scale, scale, scale),
					//		RZ::Material(rand() % 2, 0.0f, 0.0f, 0.0f, 0.0f)),
					//	//1.0f, Graphics::Color(0xFF, 0x40, 0x40, 0x00)));
					//	1.0f, Graphics::Color(0x80, 0x80, 0x80, 0x00)));
				}
			}
		}

		/*mr_world.GetSpheres().CreateObject(RZ::ConStruct<RZ::Sphere>(
			RZ::ConStruct<RZ::RenderObject>(
				RZ::ConStruct<RZ::WorldObject>(L"sphere_bounding"),
				Math::vec3<float>(0.0f, 0.0f, 0.0f),
				Math::vec3<float>(0.0f, 0.0f, 0.0f),
				Math::vec3<float>(0.0f, 0.0f, 0.0f),
				Math::vec3<float>(10.0f, 10.0f, 10.0f),
				RZ::Material(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)),
			1.0f, Graphics::Color(0xFF, 0x40, 0x40, 0x00)));*/


		CreateRoom(&mr_world);

		CreateTessellatedSphere(&mr_world,
			RZ::ConStruct<RZ::Mesh>(
				RZ::ConStruct<RZ::RenderObject>(
					RZ::ConStruct<RZ::WorldObject>(
						L"tess sphere"),
					Math::vec3<float>(0.0f, 1.0f, 0.0f),
					Math::vec3<float>(0.0f, 0.0f, 0.0f),
					Math::vec3<float>(0.0f, 0.0f, 0.0f),
					Math::vec3<float>(1.0f, 1.0f, 1.0f),
					RZ::Material(1.0f))), 10u);

		/*mr_world.GetSpheres().CreateObject(RZ::ConStruct<RZ::Sphere>(
			RZ::ConStruct<RZ::RenderObject>(
				RZ::ConStruct<RZ::WorldObject>(L"light sphere"),
				Math::vec3<float>(3.0f, 3.0f, -3.0f),
				Math::vec3<float>(0.0f, 0.0f, 0.0f),
				Math::vec3<float>(0.0f, 0.0f, 0.0f),
				Math::vec3<float>(1.0f, 1.0f, 1.0f),
				RZ::Material(0.0f, 0.0f, 0.0f, 0.0f, 100.0f)),
			1.0f, Graphics::Color(0xFF, 0xFF, 0xFF, 0x00)));*/
	}
	Scene::~Scene()
	{
	}

	void Scene::Render()
	{
		mr_engine.RenderWorld();
	}
	const Graphics::Bitmap& Scene::GetRender()
	{
		return mp_camera->GetBitmap();
	}
	void Scene::ResizeRender(size_t width, size_t height)
	{
		mp_camera->Resize(width, height);
	}
	
	void Scene::CreateCube(RZ::World* world, const RZ::ConStruct<RZ::Mesh>& conStruct)
	{
		if (world == nullptr) return;

		// create mesh
		RZ::Mesh* mesh = world->GetMeshes().CreateObject(conStruct);

		// vertices
		auto& mesh_data = mesh->GetMeshStructure();
		mesh_data.CreateVertex(-1.0f, 1.0f, -1.0f);
		mesh_data.CreateVertex(1.0f, 1.0f, -1.0f);
		mesh_data.CreateVertex(1.0f, 1.0f, 1.0f);
		mesh_data.CreateVertex(-1.0f, 1.0f, 1.0f);
		mesh_data.CreateVertex(-1.0f, -1.0f, -1.0f);
		mesh_data.CreateVertex(1.0f, -1.0f, -1.0f);
		mesh_data.CreateVertex(1.0f, -1.0f, 1.0f);
		mesh_data.CreateVertex(-1.0f, -1.0f, 1.0f);

		// texcrds
		mesh_data.CreateTexcrd(0.0f, 0.0f);
		mesh_data.CreateTexcrd(1.0f, 0.0f);
		mesh_data.CreateTexcrd(0.0f, 1.0f);
		mesh_data.CreateTexcrd(1.0f, 1.0f);


		// ~~~~ triangles ~~~~ //
		auto& vertices = mesh_data.GetVertices();
		auto& texcrds = mesh_data.GetTexcrds();

		// front
		mesh_data.CreateTriangle(
			0, 1, 4,
			0, 1, 2);
		mesh_data.CreateTriangle(
			5, 4, 1,
			3, 2, 1);

		// right
		mesh_data.CreateTriangle(
			1, 2, 5,
			0, 1, 2);
		mesh_data.CreateTriangle(
			6, 5, 2,
			3, 2, 1);

		// back
		mesh_data.CreateTriangle(
			2, 3, 6,
			0, 1, 2);
		mesh_data.CreateTriangle(
			7, 6, 3,
			3, 2, 1);

		// left
		mesh_data.CreateTriangle(
			3, 0, 7,
			0, 1, 2);
		mesh_data.CreateTriangle(
			4, 7, 0,
			3, 2, 1);

		// top
		mesh_data.CreateTriangle(
			1, 0, 2,
			3, 2, 1);
		mesh_data.CreateTriangle(
			3, 2, 0,
			0, 1, 2);

		mesh_data.CreateTriangle(
			5, 6, 4,
			3, 2, 1);
		mesh_data.CreateTriangle(
			7, 4, 6,
			0, 1, 2);


		// triangles colors
		unsigned char lc = 0x44;
		unsigned char hc = 0xFF;
		std::vector<Graphics::Color> colors{
			Graphics::Color(hc, lc, lc),
			Graphics::Color(lc, hc, lc),
			Graphics::Color(lc, lc, hc),
			Graphics::Color(hc, hc, lc),
			Graphics::Color(hc, lc, hc),
			Graphics::Color(lc, hc, hc) };

		auto& triangles = mesh_data.GetTriangles();
		for (unsigned int i = 0; i < triangles.GetCount() / 2; i++)
		{
			//mesh->Triangles[2 * i]->Color(colors[i]);
			//mesh->Triangles[2 * i + 1]->Color(colors[i]);
			triangles[2 * i].color = Graphics::Color(0x40, 0xFF, 0xFF, 0x00);
			triangles[2 * i + 1].color = Graphics::Color(0x40, 0xFF, 0xFF, 0x00);
		}

		//RayZath::Texture t(GenerateBitmap(), RayZath::Texture::FilterMode::Point);
		//mesh->LoadTexture(t);
	}
	void Scene::CreateRoom(RZ::World* world)
	{
		RZ::Mesh* mesh = world->GetMeshes().CreateObject(RZ::ConStruct<RZ::Mesh>(
			RZ::ConStruct<RZ::RenderObject>(
				RZ::ConStruct<RZ::WorldObject>(L"Room"),
				Math::vec3<float>(0.0f, 0.0f, 0.0f),
				Math::vec3<float>(0.0f, 0.0f, 0.0f),
				Math::vec3<float>(0.0f, 1.0f, 0.0f),
				Math::vec3<float>(6.0f, 3.0f, 6.0f),
				RZ::Material(0.5f, 0.001f)),
			8u, 12u, 18u));

		std::vector<Math::vec3<float>*> vertices;

		// vertices
		mesh->GetMeshStructure().CreateVertex(-1.0f, 1.0f, -1.0f);
		mesh->GetMeshStructure().CreateVertex(1.0f, 1.0f, -1.0f);
		mesh->GetMeshStructure().CreateVertex(1.0f, 1.0f, 1.0f);
		mesh->GetMeshStructure().CreateVertex(-1.0f, 1.0f, 1.0f);
		mesh->GetMeshStructure().CreateVertex(-1.0f, -1.0f, -1.0f);
		mesh->GetMeshStructure().CreateVertex(1.0f, -1.0f, -1.0f);
		mesh->GetMeshStructure().CreateVertex(1.0f, -1.0f, 1.0f);
		mesh->GetMeshStructure().CreateVertex(-1.0f, -1.0f, 1.0f);

		// texture coordinates
		//mesh->Texcrds.CreateTexcd(0.0f, 1.0f);
		//mesh->Texcrds.CreateTexcd(1.0f, 1.0f);
		//mesh->Texcrds.CreateTexcd(1.0f, 0.0f);
		//mesh->Texcrds.CreateTexcd(0.0f, 0.0f);

		// texture coordinates
		for (int i = 0; i <= 8; i++)
		{
			mesh->GetMeshStructure().CreateTexcrd(i / 8.0f, 0.0f);
			mesh->GetMeshStructure().CreateTexcrd(i / 8.0f, 1.0f);
		}

		//// texture bitmap
		//mesh->LoadTexture(RZ::Texture(GenerateBitmap(), RZ::Texture::FilterMode::Point));
		mesh->LoadTexture(RZ::Texture(GenerateColorBitmap(), RZ::Texture::FilterMode::Point));


		//// [>] Creation and Description of each triangle
		///// floor
		//mesh->Triangles.CreateTriangle(vertices[4], vertices[7], vertices[6], mesh->Texcrds[0], mesh->Texcrds[3], mesh->Texcrds[2]);
		//mesh->Triangles.CreateTriangle(vertices[4], vertices[6], vertices[5], mesh->Texcrds[0], mesh->Texcrds[2], mesh->Texcrds[1]);
		///// ceil
		//mesh->Triangles.CreateTriangle(vertices[0], vertices[2], vertices[3], mesh->Texcrds[0], mesh->Texcrds[2], mesh->Texcrds[3]);
		//mesh->Triangles.CreateTriangle(vertices[0], vertices[1], vertices[2], mesh->Texcrds[0], mesh->Texcrds[1], mesh->Texcrds[2]);
		///// left wall
		//mesh->Triangles.CreateTriangle(vertices[0], vertices[3], vertices[7], mesh->Texcrds[0], mesh->Texcrds[3], mesh->Texcrds[2]);
		//mesh->Triangles.CreateTriangle(vertices[0], vertices[7], vertices[4], mesh->Texcrds[0], mesh->Texcrds[2], mesh->Texcrds[1]);
		///// right wall
		//mesh->Triangles.CreateTriangle(vertices[1], vertices[6], vertices[2], mesh->Texcrds[0], mesh->Texcrds[2], mesh->Texcrds[3]);
		//mesh->Triangles.CreateTriangle(vertices[1], vertices[5], vertices[6], mesh->Texcrds[0], mesh->Texcrds[1], mesh->Texcrds[2]);
		///// back wall
		//mesh->Triangles.CreateTriangle(vertices[3], vertices[2], vertices[6], mesh->Texcrds[0], mesh->Texcrds[3], mesh->Texcrds[2]);
		//mesh->Triangles.CreateTriangle(vertices[3], vertices[6], vertices[7], mesh->Texcrds[0], mesh->Texcrds[2], mesh->Texcrds[1]);
		///// front wall
		//mesh->Triangles.CreateTriangle(vertices[0], vertices[5], vertices[1], mesh->Texcrds[0], mesh->Texcrds[2], mesh->Texcrds[3]);
		//mesh->Triangles.CreateTriangle(vertices[0], vertices[4], vertices[5], mesh->Texcrds[0], mesh->Texcrds[1], mesh->Texcrds[2]);


		// [>] Creation and Description of each triangle
		/// floor
		mesh->GetMeshStructure().CreateTriangle(4, 7, 6, 1, 0, 2);
		mesh->GetMeshStructure().CreateTriangle(4, 6, 5, 1, 2, 3);
		///// ceil
		//mesh->Triangles.CreateTriangle(vertices[0], vertices[2], vertices[3], mesh->Texcrds[2], mesh->Texcrds[5], mesh->Texcrds[3]);
		//mesh->Triangles.CreateTriangle(vertices[0], vertices[1], vertices[2], mesh->Texcrds[2], mesh->Texcrds[4], mesh->Texcrds[5]);
		//// left wall
		//mesh->Triangles.CreateTriangle(vertices[0], vertices[3], vertices[7], mesh->Texcrds[4], mesh->Texcrds[6], mesh->Texcrds[7]);
		//mesh->Triangles.CreateTriangle(vertices[0], vertices[7], vertices[4], mesh->Texcrds[4], mesh->Texcrds[7], mesh->Texcrds[5]);
		//// right wall
		//mesh->Triangles.CreateTriangle(vertices[1], vertices[6], vertices[2], mesh->Texcrds[8], mesh->Texcrds[7], mesh->Texcrds[6]);
		//mesh->Triangles.CreateTriangle(vertices[1], vertices[5], vertices[6], mesh->Texcrds[8], mesh->Texcrds[9], mesh->Texcrds[7]);
		//// back wall
		//mesh->Triangles.CreateTriangle(vertices[3], vertices[2], vertices[6], mesh->Texcrds[8], mesh->Texcrds[10], mesh->Texcrds[11]);
		//mesh->Triangles.CreateTriangle(vertices[3], vertices[6], vertices[7], mesh->Texcrds[8], mesh->Texcrds[11], mesh->Texcrds[9]);
		///// front wall
		//mesh->Triangles.CreateTriangle(vertices[0], vertices[5], vertices[1], mesh->Texcrds[12], mesh->Texcrds[11], mesh->Texcrds[10]);
		//mesh->Triangles.CreateTriangle(vertices[0], vertices[4], vertices[5], mesh->Texcrds[12], mesh->Texcrds[13], mesh->Texcrds[11]);

		using namespace Graphics;
		// floor
		mesh->GetMeshStructure().GetTriangles()[0].color = Color(0x43, 0x8A, 0x6E);
		mesh->GetMeshStructure().GetTriangles()[1].color = Color(0x43, 0x8A, 0x6E);
		//// ceil
		//mesh->Triangles[2]->Color(Color(0xFF, 0xFF, 0xFF));
		//mesh->Triangles[3]->Color(Color(0xFF, 0xFF, 0xFF));
		//// left wall
		//mesh->Triangles[4]->Color(Color(0xFF, 0x22, 0x22));
		//mesh->Triangles[5]->Color(Color(0xFF, 0x22, 0x22));
		//// right wall
		//mesh->Triangles[6]->Color(Color(0x22, 0xFF, 0x22));
		//mesh->Triangles[7]->Color(Color(0x22, 0xFF, 0x22));
		//// back wall
		//mesh->Triangles[8]->Color(Color(0xFF, 0xFF, 0x44));
		//mesh->Triangles[9]->Color(Color(0xFF, 0xFF, 0x44));
		////// front wall
		//mesh->Triangles[10]->Color(Color(0x44, 0xFF, 0xFF));
		//mesh->Triangles[11]->Color(Color(0x44, 0xFF, 0xFF));
	}

	void Scene::CreateTessellatedSphere(
		RZ::World* world,
		const RZ::ConStruct<RZ::Mesh>& conStruct,
		const size_t& res)
	{
		// [>] create Mesh object
		RZ::Mesh* mesh = world->GetMeshes().CreateObject(conStruct);
		mesh->GetMeshStructure().Reset(5000u, 2u, 5000u);

		// [>] Create vertices
		// middle layer vertices
		for (size_t i = 1; i < res - 1; ++i)
		{
			for (size_t j = 0; j < res; ++j)
			{
				RZ::Vertex v(0.0f, 1.0f, 0.0f);
				v.RotateX(((float)i / (float)(res - 1)) * Math::constants<float>::Pi);
				v.RotateY(((float)j / (float)(res)) * Math::constants<float>::Pi * 2.0f);
				mesh->GetMeshStructure().CreateVertex(v);
			}
		}

		// top and bottom vertices
		Math::vec3<float>* vTop = mesh->GetMeshStructure().CreateVertex(0.0f, 1.0f, 0.0f);
		Math::vec3<float>* vBottom = mesh->GetMeshStructure().CreateVertex(0.0f, -1.0f, 0.0f);


		// [>] Create triangles
		// hat and foot
		for (size_t i = 0; i < res; ++i)
		{
			mesh->GetMeshStructure().CreateTriangle(
				vTop, 
				&mesh->GetMeshStructure().GetVertices()[(i + 1) % res],
				&mesh->GetMeshStructure().GetVertices()[i]);
			mesh->GetMeshStructure().CreateTriangle(
				vBottom,
				&mesh->GetMeshStructure().GetVertices()[res * (res - 3) + (i + 1) % res],
				&mesh->GetMeshStructure().GetVertices()[res * (res - 3) + i % res]);
		}

		// middle layers
		for (size_t i = 0; i < res - 3; ++i)
		{
			for (size_t j = 0; j < res; ++j)
			{
				mesh->GetMeshStructure().CreateTriangle(
					&mesh->GetMeshStructure().GetVertices()[i * res + j], 
					&mesh->GetMeshStructure().GetVertices()[(i + 1) * res + (j + 1) % res], 
					&mesh->GetMeshStructure().GetVertices()[(i + 1) * res + j]);

				mesh->GetMeshStructure().CreateTriangle(
					&mesh->GetMeshStructure().GetVertices()[i * res + j], 
					&mesh->GetMeshStructure().GetVertices()[i * res + (j + 1) % res], 
					&mesh->GetMeshStructure().GetVertices()[(i + 1) * res + (j + 1) % res]);				
			}
		}


		// triangle coloring
		for (size_t i = 0; i < mesh->GetMeshStructure().GetTriangles().GetCapacity(); ++i)
		{
			mesh->GetMeshStructure().GetTriangles()[i].color = 
				Graphics::Color(
					rand() % 63 + 192,
					rand() % 63 + 192, 
					rand() % 63 + 192,
					0x00);
		}
	}
}