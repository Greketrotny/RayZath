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
	unsigned int resolution = 64;
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
		// cameras
		mp_camera = mr_world.GetCameras().CreateObject(RZ::ConStruct<RZ::Camera>(
			RZ::ConStruct<RZ::WorldObject>(L"camera 1"),
			Math::vec3<float>(0.0f, 5.0f, -10.0f),
			Math::vec3<float>(-0.1f, 0.0f, 0.0f),
			1200, 700,
			Math::angle<Math::deg, float>(100.0f),
			10.0f, 0.001f, true));

		//// point lights
		//mr_world.GetPointLights().CreateObject(RZ::ConStruct<RZ::PointLight>(
		//	RZ::ConStruct<RZ::WorldObject>(L"point light 1"),
		//	Math::vec3<float>(0.0f, 3.0f, 0.0f),
		//	Graphics::Color(0xFF, 0xFF, 0xFF),
		//	0.2f, 100.0f));

		// light sphere
		RZ::Sphere* s3 = mr_world.GetSpheres().CreateObject(RZ::ConStruct<RZ::Sphere>(
			RZ::ConStruct<RZ::RenderObject>(
				RZ::ConStruct<RZ::WorldObject>(L"light sphere"),
				Math::vec3<float>(-5.0f, 5.0f, 0.0f),
				Math::vec3<float>(0.0f, 0.0f, -0.7f),
				Math::vec3<float>(0.0f, 0.0f, 0.0f),
				Math::vec3<float>(1.0f, 0.1f, 5.0f),
				RZ::Material(0.0f, 0.0f, 0.0f, 0.0f, 50.0f))));
		s3->SetColor(Graphics::Color(0xFF, 0xFF, 0x40));
		// light sphere 2
		RZ::Sphere* s4 = mr_world.GetSpheres().CreateObject(RZ::ConStruct<RZ::Sphere>(
			RZ::ConStruct<RZ::RenderObject>(
				RZ::ConStruct<RZ::WorldObject>(L"light sphere 2"),
				Math::vec3<float>(5.0f, 5.0f, 0.0f),
				Math::vec3<float>(0.0f, 0.0f, 0.7f),
				Math::vec3<float>(0.0f, 0.0f, 0.0f),
				Math::vec3<float>(1.0f, 0.1f, 5.0f),
				RZ::Material(0.0f, 0.0f, 0.0f, 0.0f, 50.0f))));
		s4->SetColor(Graphics::Color(0x40, 0xFF, 0xFF));


		CreateRoom(&mr_world);

		/*size_t res = 8u;
		CreateTessellatedSphere(
			&mr_world, 
			RZ::ConStruct<RZ::Mesh>(
				RZ::ConStruct<RZ::RenderObject>(
					RZ::ConStruct<RZ::WorldObject>(L"tes mesh 1"),
					Math::vec3<float>(-2.0f, 1.0f, 2.0f),
					Math::vec3<float>(0.0f, 0.0f, 0.0f),
					Math::vec3<float>(0.0f, 0.0f, 0.0f),
					Math::vec3<float>(1.0f, 1.0f, 1.0f),
					RZ::Material(0.5f)),
				res * (res - 1) + 2, res * (res - 3) * 2 + 2 * res),
			res);*/

		int n = 4;
		for (int x = -n/2; x <= n/2; x++)
		{
			for (int y = -n/2; y <= n/2; y++)
			{
				for (int z = -n/2; z <= n/2; z++)
				{
					RZ::Sphere* s = mr_world.GetSpheres().CreateObject(RZ::ConStruct<RZ::Sphere>(
						RZ::ConStruct<RZ::RenderObject>(
							RZ::ConStruct<RZ::WorldObject>(L"sphere"),
							Math::vec3<float>(x, y, z) * 1.5f + Math::vec3<float>(0.0f, n, 0.0f),
							Math::vec3<float>(0.0f, 0.0f, 0.0f),
							Math::vec3<float>(0.0f, 0.0f, 0.0f),
							Math::vec3<float>(0.5f, 0.5f, 0.5f),
							RZ::Material(0.0f, 0.0f, 0.0f, 0.0f, 0.0f))));
					s->SetColor(Graphics::Color(0xFF, 0xFF, 0xFF));
				}
			}
		}
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
		mesh->Vertices.CreateVertex(-1.0f, 1.0f, -1.0f);
		mesh->Vertices.CreateVertex(1.0f, 1.0f, -1.0f);
		mesh->Vertices.CreateVertex(1.0f, 1.0f, 1.0f);
		mesh->Vertices.CreateVertex(-1.0f, 1.0f, 1.0f);
		mesh->Vertices.CreateVertex(-1.0f, -1.0f, -1.0f);
		mesh->Vertices.CreateVertex(1.0f, -1.0f, -1.0f);
		mesh->Vertices.CreateVertex(1.0f, -1.0f, 1.0f);
		mesh->Vertices.CreateVertex(-1.0f, -1.0f, 1.0f);

		// texcrds
		mesh->Texcrds.CreateTexcd(0.0f, 0.0f);
		mesh->Texcrds.CreateTexcd(1.0f, 0.0f);
		mesh->Texcrds.CreateTexcd(0.0f, 1.0f);
		mesh->Texcrds.CreateTexcd(1.0f, 1.0f);


		// ~~~~ triangles ~~~~ //
		// front
		mesh->Triangles.CreateTriangle(
			mesh->Vertices[0], mesh->Vertices[1], mesh->Vertices[4],
			mesh->Texcrds[0], mesh->Texcrds[1], mesh->Texcrds[2]);
		mesh->Triangles.CreateTriangle(
			mesh->Vertices[5], mesh->Vertices[4], mesh->Vertices[1],
			mesh->Texcrds[3], mesh->Texcrds[2], mesh->Texcrds[1]);

		// right
		mesh->Triangles.CreateTriangle(
			mesh->Vertices[1], mesh->Vertices[2], mesh->Vertices[5],
			mesh->Texcrds[0], mesh->Texcrds[1], mesh->Texcrds[2]);
		mesh->Triangles.CreateTriangle(
			mesh->Vertices[6], mesh->Vertices[5], mesh->Vertices[2],
			mesh->Texcrds[3], mesh->Texcrds[2], mesh->Texcrds[1]);

		// back
		mesh->Triangles.CreateTriangle(
			mesh->Vertices[2], mesh->Vertices[3], mesh->Vertices[6],
			mesh->Texcrds[0], mesh->Texcrds[1], mesh->Texcrds[2]);
		mesh->Triangles.CreateTriangle(
			mesh->Vertices[7], mesh->Vertices[6], mesh->Vertices[3],
			mesh->Texcrds[3], mesh->Texcrds[2], mesh->Texcrds[1]);

		// left
		mesh->Triangles.CreateTriangle(
			mesh->Vertices[3], mesh->Vertices[0], mesh->Vertices[7],
			mesh->Texcrds[0], mesh->Texcrds[1], mesh->Texcrds[2]);
		mesh->Triangles.CreateTriangle(
			mesh->Vertices[4], mesh->Vertices[7], mesh->Vertices[0],
			mesh->Texcrds[3], mesh->Texcrds[2], mesh->Texcrds[1]);

		// top
		mesh->Triangles.CreateTriangle(
			mesh->Vertices[1], mesh->Vertices[0], mesh->Vertices[2],
			mesh->Texcrds[3], mesh->Texcrds[2], mesh->Texcrds[1]);
		mesh->Triangles.CreateTriangle(
			mesh->Vertices[3], mesh->Vertices[2], mesh->Vertices[0],
			mesh->Texcrds[0], mesh->Texcrds[1], mesh->Texcrds[2]);

		mesh->Triangles.CreateTriangle(
			mesh->Vertices[5], mesh->Vertices[6], mesh->Vertices[4],
			mesh->Texcrds[3], mesh->Texcrds[2], mesh->Texcrds[1]);
		mesh->Triangles.CreateTriangle(
			mesh->Vertices[7], mesh->Vertices[4], mesh->Vertices[6],
			mesh->Texcrds[0], mesh->Texcrds[1], mesh->Texcrds[2]);


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

		for (unsigned int i = 0; i < mesh->Triangles.Count / 2; i++)
		{
			//mesh->Triangles[2 * i]->Color(colors[i]);
			//mesh->Triangles[2 * i + 1]->Color(colors[i]);
			mesh->Triangles[2 * i]->Color(Graphics::Color(0x40, 0xFF, 0xFF, 0x00));
			mesh->Triangles[2 * i + 1]->Color(Graphics::Color(0x40, 0xFF, 0xFF, 0x00));
		}

		//RayZath::Texture t(GenerateBitmap(), RayZath::Texture::FilterMode::Point);
		//mesh->LoadTexture(t);

		mesh->TransposeComponents();
		mr_world.RequestUpdate();
	}
	void Scene::CreateRoom(RZ::World* world)
	{
		RZ::Mesh* mesh = world->GetMeshes().CreateObject(RZ::ConStruct<RZ::Mesh>(
			RZ::ConStruct<RZ::RenderObject>(
				RZ::ConStruct<RZ::WorldObject>(L"Room"),
				Math::vec3<float>(0.0f, 3.0f, 0.0f),
				Math::vec3<float>(0.0f, 0.0f, 0.0f),
				Math::vec3<float>(0.0f, 3.0f, 0.0f),
				Math::vec3<float>(6.0f, 3.0f, 6.0f),
				RZ::Material(0.5f, 0.001f)),
			8u, 12u, 18u));

		std::vector<Math::vec3<float>*> vertices;

		// vertices
		vertices.push_back(mesh->Vertices.CreateVertex(-1.0f, 1.0f, -1.0f));
		vertices.push_back(mesh->Vertices.CreateVertex(1.0f, 1.0f, -1.0f));
		vertices.push_back(mesh->Vertices.CreateVertex(1.0f, 1.0f, 1.0f));
		vertices.push_back(mesh->Vertices.CreateVertex(-1.0f, 1.0f, 1.0f));
		vertices.push_back(mesh->Vertices.CreateVertex(-1.0f, -1.0f, -1.0f));
		vertices.push_back(mesh->Vertices.CreateVertex(1.0f, -1.0f, -1.0f));
		vertices.push_back(mesh->Vertices.CreateVertex(1.0f, -1.0f, 1.0f));
		vertices.push_back(mesh->Vertices.CreateVertex(-1.0f, -1.0f, 1.0f));

		// texture coordinates
		//mesh->Texcrds.CreateTexcd(0.0f, 1.0f);
		//mesh->Texcrds.CreateTexcd(1.0f, 1.0f);
		//mesh->Texcrds.CreateTexcd(1.0f, 0.0f);
		//mesh->Texcrds.CreateTexcd(0.0f, 0.0f);

		// texture coordinates
		for (int i = 0; i <= 8; i++)
		{
			mesh->Texcrds.CreateTexcd(i / 8.0f, 0.0f);
			mesh->Texcrds.CreateTexcd(i / 8.0f, 1.0f);
		}

		//// texture bitmap
		//mesh->LoadTexture(RZ::Texture(GenerateBitmap(), RZ::Texture::FilterMode::Point));
		//mesh->LoadTexture(RZ::Texture(GenerateColorBitmap(), RZ::Texture::FilterMode::Point));


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
		mesh->Triangles.CreateTriangle(vertices[4], vertices[7], vertices[6], mesh->Texcrds[1], mesh->Texcrds[0], mesh->Texcrds[2]);
		mesh->Triangles.CreateTriangle(vertices[4], vertices[6], vertices[5], mesh->Texcrds[1], mesh->Texcrds[2], mesh->Texcrds[3]);
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
		mesh->Triangles[0]->Color(Color(0x43, 0x8A, 0x6E));
		mesh->Triangles[1]->Color(Color(0x43, 0x8A, 0x6E));
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

		mesh->TransposeComponents();
		mr_world.RequestUpdate();
	}

	void Scene::CreateTessellatedSphere(
		RZ::World* world,
		const RZ::ConStruct<RZ::Mesh>& conStruct,
		const size_t& res)
	{
		// [>] create Mesh object
		RZ::Mesh* mesh = world->GetMeshes().CreateObject(conStruct);


		// [>] Create vertices
		// middle layer vertices
		for (size_t i = 1; i < res - 1; ++i)
		{
			for (size_t j = 0; j < res; ++j)
			{
				RZ::Mesh::Vertex* v = mesh->Vertices.CreateVertex(0.0f, 1.0f, 0.0f);
				v->RotateX(((float)i / (float)(res - 1)) * Math::constants<float>::Pi);
				v->RotateY(((float)j / (float)(res)) * Math::constants<float>::Pi * 2.0f);
			}
		}

		// top and bottom vertices
		RZ::Mesh::Vertex* vTop = mesh->Vertices.CreateVertex(0.0f, 1.0f, 0.0f);
		RZ::Mesh::Vertex* vBottom = mesh->Vertices.CreateVertex(0.0f, -1.0f, 0.0f);


		// [>] Create triangles
		// hat and foot
		for (size_t i = 0; i < res; ++i)
		{
			mesh->Triangles.CreateTriangle(vTop, mesh->Vertices[(i + 1) % res], mesh->Vertices[i]);
			mesh->Triangles.CreateTriangle(
				vBottom,
				mesh->Vertices[res * (res - 3) + i % res],
				mesh->Vertices[res * (res - 3) + (i - 1) % res]);
		}

		// middle layers
		for (size_t i = 0; i < res - 3; ++i)
		{
			for (size_t j = 0; j < res; ++j)
			{
				mesh->Triangles.CreateTriangle(
					mesh->Vertices[i * res + j], 
					mesh->Vertices[(i + 1) * res + (j + 1) % res], 
					mesh->Vertices[(i + 1) * res + j]);

				mesh->Triangles.CreateTriangle(
					mesh->Vertices[i * res + j], 
					mesh->Vertices[i * res + (j + 1) % res], 
					mesh->Vertices[(i + 1) * res + (j + 1) % res]);				
			}
		}


		// triangle coloring
		for (size_t i = 0; i < mesh->Triangles.Capacity; ++i)
		{
			mesh->Triangles[i]->Color(Graphics::Color(0xFF, 0xFF, 0xFF, 0x00));
		}


		mesh->TransposeComponents();
		mr_world.RequestUpdate();
	}
}