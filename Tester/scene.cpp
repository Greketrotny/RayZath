#include "scene.h"
#include "application.h"
#include "loader.h"

Graphics::Bitmap GenerateColorBitmap()
{
	unsigned int resolution = 32;
	Graphics::Bitmap bitmap(resolution * 8, resolution);

	unsigned char hs = 0xCC;
	unsigned char ls = 0x44;
	//std::vector<Graphics::Color> colors{
	//	Graphics::Color(hs, ls, ls),	// red
	//	Graphics::Color(ls, hs, ls),	// green
	//	Graphics::Color(ls, ls, hs),	// blue
	//	Graphics::Color(ls, hs, hs),	// cyan
	//	Graphics::Color(hs, hs, ls),	// yellow
	//	Graphics::Color(hs, ls, hs),	// magenta
	//	Graphics::Color(hs, hs, hs),	// white
	//	Graphics::Color(ls, ls, ls),	// dark grey
	//};
	std::vector<Graphics::Color> colors{
		Graphics::Color(hs, hs, hs),	// white
		Graphics::Color(hs, hs, hs),	// white
		Graphics::Color(hs, ls, ls),	// red
		Graphics::Color(ls, hs, ls),	// green
		Graphics::Color(hs, hs, hs),	// white
		Graphics::Color(hs, ls, hs),	// magenta
		Graphics::Color(hs, hs, hs),	// white
		Graphics::Color(ls, ls, ls),	// dark grey
	};

	for (unsigned int i = 0; i < 8; ++i)
	{
		for (unsigned int x = 0; x < resolution; ++x)
		{
			for (unsigned int y = 0; y < resolution; ++y)
			{
				bitmap.SetPixel(resolution * i + x, y, colors[i]);
				//if ((x % 2 == 0) ^ (y % 2 == 0)) bitmap.SetPixel(resolution * i + x, y, colors[i]);
				//else bitmap.SetPixel(resolution * i + x, y, Graphics::Color::BlendAverage(colors[i], Graphics::Color(0x00, 0x00, 0x00)));

				//if (x == 2 && y == 2)
				//	bitmap.SetPixel(x, y, Graphics::Color(0x00, 0x00, 0xFF));
			}
		}
	}
	
	return bitmap;
}

Graphics::Bitmap GenerateBitmap(
	const uint32_t resolution, 
	const Graphics::Color& color1,
	const Graphics::Color& color2)
{
	Graphics::Bitmap bitmap(resolution, resolution);

	for (unsigned int x = 0; x < resolution; ++x)
	{
		for (unsigned int y = 0; y < resolution; ++y)
		{
			if ((x % 2 == 0) ^ (y % 2 == 0)) 
				bitmap.SetPixel(x, y, color1);
			else 
				bitmap.SetPixel(x, y, color2);
		}
	}
	return bitmap;
}


namespace Tester
{
	Scene::Scene(Application& app)
		: mr_app(app)
		, mr_engine(RZ::Engine::GetInstance())
		, mr_world(mr_engine.GetWorld())
	{
		// cameras
		m_camera = mr_world.GetCameras().Create(
			RZ::ConStruct<RZ::Camera>(
				L"camera 1",
				Math::vec3f(0.0f, 3.0f, -11.0f),
				Math::vec3f(0.0f, 0.0f, 0.0f),
				/*Math::vec3f(-2.0f, -4.0f, -14.0f),
				Math::vec3f(0.5f, -0.4f, 0.0f),*/
				1200u, 700u,
				Math::angle_degf(100.0f),
				10.0f, 0.001f, true));

		RZ::World& world = RZ::Engine::GetInstance().GetWorld();

		// lights
		/*RZ::Handle<RZ::PointLight> point_light1 = world.GetPointLights().Create(
			RZ::ConStruct<RZ::PointLight>(
				L"point light 1",
				Math::vec3f(4.0f, 6.0f, -4.0f),
				Graphics::Color::White,
				0.2f, 200.0f));*/
		/*world.GetSpotLights().Create(
			RZ::ConStruct<RZ::SpotLight>(
				L"spotlight 1",
				Math::vec3f(0.0f, 4.0f, -4.0f),
				Math::vec3f(0.0f, -1.0f, 1.0f),
				Graphics::Color::Red,
				0.25f, 50.0f, 0.3f, 0.5f));*/
		/*mr_world.GetDirectLights().Create(
			RZ::ConStruct<RZ::DirectLight>(
				L"direct light 1",
				Math::vec3f(1.0f, -1.0f, 1.0f),
				Graphics::Color(0xFF, 0xFF, 0xFF, 0xFF),
				10.0f, 0.02f));*/


		const int res = 0;
		const float scale = 1.0f;
		const float size = 0.2f;
		const float emitance = 200.0f / ((res + 1) * (res + 1));
		for (int i = -res; i <= res; i++)
		{
			for (int j = -res; j <= res; j++)
			{
				/*world.GetSpotLights().Create(
					RZ::ConStruct<RZ::SpotLight>(
						L"spot light " + std::to_wstring(i * res + j),
						Math::vec3f(i * scale, 6.0f, j * scale),
						Math::vec3f(0.0f, -1.0f, 0.0f),
						Graphics::Color::White,
						size, emitance, 0.3f, 0.5f));*/

				/*RZ::Handle<RZ::PointLight> point_light1 = world.GetPointLights().Create(
					RZ::ConStruct<RZ::PointLight>(
						L"point light " + std::to_wstring(i * res + j),
						Math::vec3f(i * scale, 6.0f, j * scale),
						Graphics::Color::White,
						size, emitance));*/
			}
		}

		// textures
		RZ::Handle<RZ::Texture> texture1 = world.GetTextures().Create(
			RZ::ConStruct<RZ::Texture>(
				L"texture 1",
				GenerateBitmap(
					8,
					Graphics::Color(0xFF, 0xFF, 0xFF, 0x00),
					Graphics::Color(0x80, 0x80, 0x80, 0x00)),
				RZ::Texture::FilterMode::Point));
		RZ::Handle<RZ::Texture> texture2 = world.GetTextures().Create(
			RZ::ConStruct<RZ::Texture>(
				L"texture 2",
				GenerateColorBitmap(),
				RZ::Texture::FilterMode::Point));
		RZ::Handle<RZ::Texture> env_texture = world.GetTextures().Create(
			RZ::ConStruct<RZ::Texture>(
				L"environment",
				LoadFromFile(
					"D:/Users/Greketrotny/Programming/Projects/C++/RayZath/Tester/Resources/img/environment.jpg"),
				RZ::Texture::FilterMode::Linear));

		// world
		//world.GetMaterial().SetTexture(env_texture);
		world.GetDefaultMaterial().SetColor(Graphics::Color::Green);
		//world.GetMaterial().SetEmittance(5.0f);


		// materials
		RZ::Handle<RZ::Material> mat_diffuse = world.GetMaterials().Create(
			RZ::ConStruct<RZ::Material>(
				Graphics::Color(0xC0, 0xC0, 0xC0, 0x00),
				0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
				texture1));
		RZ::Handle<RZ::Material> mat_diffuse2 = world.GetMaterials().Create(
			RZ::ConStruct<RZ::Material>(
				Graphics::Color(0xC0, 0xC0, 0xC0, 0x00),
				0.0f, 0.0f, 0.0f, 1.5f, 0.0f, 0.0f,
				/*RZ::Handle<RZ::Texture>()*/texture2));
		RZ::Handle<RZ::Material> mat_glass = world.GetMaterials().Create(
			RZ::ConStruct<RZ::Material>(
				Graphics::Color(0xFF, 0xFF, 0xFF, 0x00),
				0.0f, 0.0f, 1.0f, 1.5f, 0.0f, 0.0f));
		RZ::Handle<RZ::Material> mat_mirror = world.GetMaterials().Create(
			RZ::ConStruct<RZ::Material>(
				Graphics::Color(0xFF, 0xFF, 0xFF, 0x00),
				1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f));
		RZ::Handle<RZ::Material> mat_gloss = world.GetMaterials().Create(
			RZ::ConStruct<RZ::Material>(
				Graphics::Color(0xFF, 0xFF, 0xFF, 0x00),
				0.5f, 0.001f, 0.0f, 1.0f, 0.0f, 0.0f,
				texture1));


		// spheres
		RZ::Handle<RZ::Sphere> sphere = world.GetSpheres().Create(
			RZ::ConStruct<RZ::Sphere>(
				L"glass sphere",
				Math::vec3f(2.0f, 3.0f, -0.5f),
				Math::vec3f(0.0f, 0.0f, 0.0f),
				Math::vec3f(0.0f, 0.0f, 0.0f),
				Math::vec3f(1.0f, 1.0f, 1.0f),
				mat_glass));

		
		// cubes
		CreateCube(world, RZ::ConStruct<RZ::Mesh>(
			L"tall cube",
			Math::vec3f(-2.0f, 0.0f, 1.0f),
			Math::vec3f(
				0.0f,
				Math::angle_radf(Math::angle_degf(35.0f)).value(),
				0.0f),
			Math::vec3f(0.0f, 1.0f, 0.0f),
			Math::vec3f(1.0f, 2.0f, 1.0f),
			RZ::Handle<RZ::MeshStructure>(),
			mat_mirror));
		cube = CreateCube(world, RZ::ConStruct<RZ::Mesh>(
			L"front cube",
			Math::vec3f(2.0f, 0.0f, -0.5f),
			Math::vec3f(
				0.0f,
				Math::angle_radf(Math::angle_degf(-25.0f)).value(),
				0.0f),
			Math::vec3f(0.0f, 1.0f, 0.0f),
			Math::vec3f(1.0f, 1.0f, 1.0f),
			RZ::Handle<RZ::MeshStructure>(),
			mat_diffuse));

		// light planes
		CreateLightPlane(
			world,
			RZ::ConStruct<RZ::Mesh>(
				L"light plane",
				Math::vec3f(0.0f, 5.99f, 0.0f),
				Math::vec3f(0.0f, 0.0f, 0.0f),
				Math::vec3f(0.0f, 0.0f, 0.0f),
				Math::vec3f(1.0f, 1.0f, 1.0f)),
			Graphics::Color(0xFF, 0xFF, 0xFF));

		RZ::Handle<RZ::Mesh> room = CreateRoom(mr_world, RZ::ConStruct<RZ::Mesh>(
			L"Room",
			Math::vec3f(0.0f, 0.0f, 0.0f),
			Math::vec3f(0.0f, 0.0f, 0.0f),
			Math::vec3f(0.0f, 1.0f, 0.0f),
			Math::vec3f(5.0f, 3.0f, 3.0f),
			RZ::Handle<RZ::MeshStructure>(),
			mat_diffuse2));
		room->SetMaterial(mat_mirror, 1u);
		/*RZ::Handle<RZ::Mesh> ground = CreateGround(mr_world, RZ::ConStruct<RZ::Mesh>(
			L"ground",
			Math::vec3f(0.0f, 0.0f, 0.0f),
			Math::vec3f(0.0f, 0.0f, 0.0f),
			Math::vec3f(0.0f, 0.0f, 0.0f),
			Math::vec3f(16.0f, 1.0f, 16.0f),
			RZ::Handle<RZ::MeshStructure>(),
			mat_diffuse));*/

		// planes
		/*RZ::Handle<RZ::Plane> plane = world.GetPlanes().Create(
			RZ::ConStruct<RZ::Plane>(
				L"plane",
				Math::vec3f(0.0f, -0.1f, 0.0f),
				Math::vec3f(0.0f, 0.0f, 0.0f),
				Math::vec3f(0.0f, 0.0f, 0.0f),
				Math::vec3f(10.0f, 10.0f, 10.0f),
				mat_diffuse));*/

		//// bunny
		/*RZ::Handle<RZ::MeshStructure> bunny_structure = world.GetMeshStructures().Create(
			RZ::ConStruct<RZ::MeshStructure>());
		bunny_structure->LoadFromFile(
			L"D:/Users/Greketrotny/Programming/Projects/C++/RayZath/Tester/Resources/bunny.obj");
		bunny = world.GetMeshes().Create(
			RZ::ConStruct<RZ::Mesh>(
				L"bunny",
				Math::vec3f(0.0f, 1.0f, -2.0f),
				Math::vec3f(0.0f, 3.14f, 0.0f),
				Math::vec3f(0.0f, 0.0f, 0.0f),
				Math::vec3f(1.0f, 1.0f, 1.0f),
				bunny_structure,
				mat_diffuse));*/

		// teapot
		/*RZ::Handle<RZ::MeshStructure> teapot_structure = world.GetMeshStructures().Create(
			RZ::ConStruct<RZ::MeshStructure>());
		teapot_structure->LoadFromFile(
			L"D:/Users/Greketrotny/Programming/Projects/C++/RayZath/Tester/Resources/teapot.obj");
		this->teapot = world.GetMeshes().Create(
			RZ::ConStruct<RZ::Mesh>(
				L"teapot",
				Math::vec3f(0.0f, 1.0f, -2.0f),
				Math::vec3f(0.0f, 1.57f, 0.0f),
				Math::vec3f(0.0f, 0.0f, 0.0f),
				Math::vec3f(1.0f, 1.0f, 1.0f),
				teapot_structure,
				mat_mirror));*/

		/*CreateRoundedCube(world, RZ::ConStruct<RZ::Mesh>(
			L"rounded cube",
			Math::vec3f(0.0f, 1.2f, 0.0f),
			Math::vec3f(0.0f, 0.78f, 0.0f),
			Math::vec3f(0.0f, 0.0f, 0.0f),
			Math::vec3f(1.0f, 1.0f, 1.0f),
			RZ::Handle<RZ::MeshStructure>(),
			mat_mirror));*/
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
		return m_camera->GetBitmap();
	}
	void Scene::ResizeRender(uint32_t width, uint32_t height)
	{
		m_camera->Resize(width, height);
	}
	void Scene::Update(const float et)
	{
		/*if (mr_world.GetStateRegister().RequiresUpdate())
		{
			m_camera->LookAtPoint(
				cube->GetTransformation().GetPosition(), 
				m_camera->GetRotation().z);
		}*/
		//bunny->LookAtPoint(m_camera->GetPosition() + m_camera->GetCoordSystem().GetZAxis() * 5.0f);
		return;

		float speed = 0.001f * et;

		Math::vec3f rot = cube->GetTransformation().GetRotation();
		rot += Math::vec3f(1.0f * speed, 0.43f * speed, 0.0f);
		cube->SetRotation(rot);
	}
	
	RZ::Handle<RZ::Mesh> Scene::CreateCube(
		RZ::World& world, 
		RZ::ConStruct<RZ::Mesh> conStruct)
	{
		// create mesh structure
		RZ::Handle<RZ::MeshStructure> structure = world.GetMeshStructures().Create(
			RZ::ConStruct<RZ::MeshStructure>(8u, 4u, 0u, 12u));

		// vertices
		structure->CreateVertex(-1.0f, 1.0f, -1.0f);
		structure->CreateVertex(1.0f, 1.0f, -1.0f);
		structure->CreateVertex(1.0f, 1.0f, 1.0f);
		structure->CreateVertex(-1.0f, 1.0f, 1.0f);
		structure->CreateVertex(-1.0f, -1.0f, -1.0f);
		structure->CreateVertex(1.0f, -1.0f, -1.0f);
		structure->CreateVertex(1.0f, -1.0f, 1.0f);
		structure->CreateVertex(-1.0f, -1.0f, 1.0f);

		// texcrds
		structure->CreateTexcrd(0.0f, 0.0f);
		structure->CreateTexcrd(1.0f, 0.0f);
		structure->CreateTexcrd(0.0f, 1.0f);
		structure->CreateTexcrd(1.0f, 1.0f);

		// triangles
		auto& vertices = structure->GetVertices();
		auto& texcrds = structure->GetTexcrds();

		// front
		structure->CreateTriangle(
			&vertices[0], &vertices[1], &vertices[4],
			&texcrds[0], &texcrds[1], &texcrds[2]);
		structure->CreateTriangle(
			&vertices[5], &vertices[4], &vertices[1],
			&texcrds[3], &texcrds[2], &texcrds[1]);

		// right
		structure->CreateTriangle(
			&vertices[1], &vertices[2], &vertices[5],
			&texcrds[0], &texcrds[1], &texcrds[2]);
		structure->CreateTriangle(
			&vertices[6], &vertices[5], &vertices[2],
			&texcrds[3], &texcrds[2], &texcrds[1]);

		// back
		structure->CreateTriangle(
			&vertices[2], &vertices[3], &vertices[6],
			&texcrds[0], &texcrds[2], &texcrds[1]);
		structure->CreateTriangle(
			&vertices[7], &vertices[6], &vertices[3],
			&texcrds[3], &texcrds[2], &texcrds[1]);

		// left
		structure->CreateTriangle(
			&vertices[3], &vertices[0], &vertices[7],
			&texcrds[0], &texcrds[1], &texcrds[2]);
		structure->CreateTriangle(
			&vertices[4], &vertices[7], &vertices[0],
			&texcrds[3], &texcrds[2], &texcrds[1]);

		// top
		structure->CreateTriangle(
			&vertices[1], &vertices[0], &vertices[2],
			&texcrds[3], &texcrds[2], &texcrds[1]);
		structure->CreateTriangle(
			&vertices[3], &vertices[2], &vertices[0],
			&texcrds[0], &texcrds[1], &texcrds[2]);

		// bottom
		structure->CreateTriangle(
			&vertices[5], &vertices[6], &vertices[4],
			&texcrds[3], &texcrds[2], &texcrds[1]);
		structure->CreateTriangle(
			&vertices[7], &vertices[4], &vertices[6],
			&texcrds[0], &texcrds[1], &texcrds[2]);

		//RayZath::Texture t(GenerateBitmap(), RayZath::Texture::FilterMode::Point);
		//mesh->LoadTexture(t);

		conStruct.mesh_structure = structure;
		return world.GetMeshes().Create(conStruct);
	}
	RZ::Handle<RZ::Mesh> Scene::CreateRoom(
		RZ::World& world,
		RZ::ConStruct<RZ::Mesh> conStruct)
	{
		// [>] Create mesh structure
		conStruct.mesh_structure = world.GetMeshStructures().Create(
			RZ::ConStruct<RZ::MeshStructure>(8u, 14u, 18u, 16u));
		auto& structure = conStruct.mesh_structure;

		// vertices
		structure->CreateVertex(-1.0f, 1.0f, -1.0f);
		structure->CreateVertex(1.0f, 1.0f, -1.0f);
		structure->CreateVertex(1.0f, 1.0f, 1.0f);
		structure->CreateVertex(-1.0f, 1.0f, 1.0f);
		structure->CreateVertex(-1.0f, -1.0f, -1.0f);
		structure->CreateVertex(1.0f, -1.0f, -1.0f);
		structure->CreateVertex(1.0f, -1.0f, 1.0f);
		structure->CreateVertex(-1.0f, -1.0f, 1.0f);

		// texture coordinates
		//mesh->Texcrds.CreateTexcd(0.0f, 1.0f);
		//mesh->Texcrds.CreateTexcd(1.0f, 1.0f);
		//mesh->Texcrds.CreateTexcd(1.0f, 0.0f);
		//mesh->Texcrds.CreateTexcd(0.0f, 0.0f);
		// texture coordinates
		for (int i = 0; i <= 8; i++)
		{
			structure->CreateTexcrd(i / 8.0f, 0.0f);
			structure->CreateTexcrd(i / 8.0f, 1.0f);
		}

		//// texture bitmap
		//mesh->LoadTexture(RZ::Texture(GenerateBitmap(), RZ::Texture::FilterMode::Point));
		//mesh->LoadTexture(RZ::Texture(GenerateColorBitmap(), RZ::Texture::FilterMode::Point));
		/* main render: 66ms */

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
		auto& vertices = structure->GetVertices();
		auto& texcrds = structure->GetTexcrds();

		/// floor
		structure->CreateTriangle(
			&vertices[4], &vertices[7], &vertices[6], 
			&texcrds[1], &texcrds[0], &texcrds[2]);
		structure->CreateTriangle(
			&vertices[4], &vertices[6], &vertices[5],
			&texcrds[1], &texcrds[2], &texcrds[3]);
		//// ceil
		structure->CreateTriangle(
			&vertices[0], &vertices[2], &vertices[3],
			&texcrds[2], &texcrds[5], &texcrds[3]);
		structure->CreateTriangle(
			&vertices[0], &vertices[1], &vertices[2],
			&texcrds[2], &texcrds[4], &texcrds[5]);
		//// left wall
		structure->CreateTriangle(
			&vertices[0], &vertices[3], &vertices[7],
			&texcrds[4], &texcrds[6], &texcrds[7]);
		structure->CreateTriangle(
			&vertices[0], &vertices[7], &vertices[4],
			&texcrds[4], &texcrds[7], &texcrds[5]);
		//// right wall
		structure->CreateTriangle(
			&vertices[1], &vertices[6], &vertices[2],
			&texcrds[8], &texcrds[7], &texcrds[6]/*,
			nullptr, nullptr, nullptr,
			1u*/);
		structure->CreateTriangle(
			&vertices[1], &vertices[5], &vertices[6],
			&texcrds[8], &texcrds[9], &texcrds[7]/*,
			nullptr, nullptr, nullptr,
			1u*/);
		//// back wall
		structure->CreateTriangle(
			&vertices[3], &vertices[2], &vertices[6],
			&texcrds[8], &texcrds[10], &texcrds[11]);
		structure->CreateTriangle(
			&vertices[3], &vertices[6], &vertices[7],
			&texcrds[8], &texcrds[11], &texcrds[9]);
		/// front wall
		/*structure->CreateTriangle(
			&vertices[0], &vertices[5], &vertices[1],
			&texcrds[10], &texcrds[13], &texcrds[12]);
		structure->CreateTriangle(
			&vertices[0], &vertices[4], &vertices[5],
			&texcrds[10], &texcrds[11], &texcrds[13]);*/

		return world.GetMeshes().Create(conStruct);
	}


	RZ::Handle<RZ::Mesh> Scene::CreateGround(
		RZ::World& world,
		RZ::ConStruct<RZ::Mesh> construct)
	{
		construct.mesh_structure = world.GetMeshStructures().Create(
			RZ::ConStruct<RZ::MeshStructure>(4u, 4u, 4u, 2u));

		auto& structure = construct.mesh_structure;

		structure->CreateVertex(-1.0f, 0.0f, -1.0f);
		structure->CreateVertex(1.0f, 0.0f, -1.0f);
		structure->CreateVertex(1.0f, 0.0f, 1.0f);
		structure->CreateVertex(-1.0f, 0.0f, 1.0f);

		structure->CreateTexcrd(0.0f, 0.0f);
		structure->CreateTexcrd(1.0f, 0.0f);
		structure->CreateTexcrd(0.0f, 1.0f);
		structure->CreateTexcrd(1.0f, 1.0f);

		structure->CreateTriangle(
			&structure->GetVertices()[0],
			&structure->GetVertices()[2],
			&structure->GetVertices()[1],
			&structure->GetTexcrds()[2],
			&structure->GetTexcrds()[1], 
			&structure->GetTexcrds()[3]);
		structure->CreateTriangle(
			&structure->GetVertices()[0],
			&structure->GetVertices()[3],
			&structure->GetVertices()[2],
			&structure->GetTexcrds()[2],
			&structure->GetTexcrds()[0],
			&structure->GetTexcrds()[1]);

		return world.GetMeshes().Create(construct);
	}

	RZ::Handle<RZ::Mesh> Scene::CreateLightPlane(
		RZ::World& world,
		RZ::ConStruct<RZ::Mesh> con_struct,
		const Graphics::Color& color)
	{
		// mesh structure
		RZ::Handle<RZ::MeshStructure> structure = world.GetMeshStructures().Create(
			RZ::ConStruct<RZ::MeshStructure>(4u, 0u, 0u, 2u));

		structure->CreateVertex(-1.0f, 0.0f, -1.0f);
		structure->CreateVertex(1.0f, 0.0f, -1.0f);
		structure->CreateVertex(1.0f, 0.0f, 1.0f);
		structure->CreateVertex(-1.0f, 0.0f, 1.0f);

		structure->CreateTriangle(
			&structure->GetVertices()[0],
			&structure->GetVertices()[1],
			&structure->GetVertices()[2]);
		structure->CreateTriangle(
			&structure->GetVertices()[0],
			&structure->GetVertices()[2],
			&structure->GetVertices()[3]);

		// material
		RZ::Handle<RZ::Material> material = world.GetMaterials().Create(
			RZ::ConStruct<RZ::Material>(
				color,
				1.0f, 0.0f, 0.0f, 1.0f, 50.0f, 0.0f));

		con_struct.material = material;
		con_struct.mesh_structure = structure;

		return world.GetMeshes().Create(con_struct);
	}

	RZ::Handle<RZ::Mesh> Scene::CreateRoundedCube(
		RZ::World& world,
		RZ::ConStruct<RZ::Mesh> con_struct)
	{
		// mesh structure
		con_struct.mesh_structure = world.GetMeshStructures().Create(
			RZ::ConStruct<RZ::MeshStructure>());
		con_struct.mesh_structure->LoadFromFile(
			L"D:/Users/Greketrotny/Programming/Projects/C++/RayZath/Tester/Resources/rounded-cube.obj");

		return world.GetMeshes().Create(con_struct);
	}
	
	//void Scene::CreateTessellatedSphere(
	//	RZ::World* world,
	//	const RZ::ConStruct<RZ::Mesh>& conStruct,
	//	const uint32_t& res)
	//{
	//	// [>] create Mesh object
	//	RZ::Mesh* mesh = world->GetMeshes().Create(conStruct);
	//	mesh->GetMeshStructure()->Reset(10000u, 2u, 10000u, 10000u);

	//	// [>] Create vertices
	//	// middle layer vertices
	//	for (uint32_t i = 1; i < res - 1; ++i)
	//	{
	//		for (uint32_t j = 0; j < res; ++j)
	//		{
	//			RZ::Vertex v(0.0f, 1.0f, 0.0f);
	//			v.RotateX(((float)i / (float)(res - 1)) * Math::constants<float>::pi);
	//			v.RotateY(((float)j / (float)(res)) * Math::constants<float>::pi * 2.0f);
	//			mesh->GetMeshStructure()->CreateVertex(v);
	//			mesh->GetMeshStructure()->CreateNormal(v);
	//		}
	//	}

	//	// top and bottom vertices
	//	Math::vec3f* vTop = mesh->GetMeshStructure()->CreateVertex(0.0f, 1.0f, 0.0f);
	//	Math::vec3f* vBottom = mesh->GetMeshStructure()->CreateVertex(0.0f, -1.0f, 0.0f);

	//	Math::vec3f* vNTop = mesh->GetMeshStructure()->CreateNormal(0.0f, 1.0f, 0.0f);
	//	Math::vec3f* vNBottom = mesh->GetMeshStructure()->CreateNormal(0.0f, -1.0f, 0.0f);


	//	// [>] Create triangles
	//	// hat and foot
	//	for (uint32_t i = 0; i < res; ++i)
	//	{
	//		mesh->GetMeshStructure()->CreateTriangle(
	//			vTop, 
	//			&mesh->GetMeshStructure()->GetVertices()[(i + 1) % res],
	//			&mesh->GetMeshStructure()->GetVertices()[i],
	//			nullptr, nullptr, nullptr,
	//			vNTop,
	//			&mesh->GetMeshStructure()->GetNormals()[(i + 1) % res],
	//			&mesh->GetMeshStructure()->GetNormals()[i]);
	//		mesh->GetMeshStructure()->CreateTriangle(
	//			vBottom,
	//			&mesh->GetMeshStructure()->GetVertices()[res * (res - 3) + i % res],
	//			&mesh->GetMeshStructure()->GetVertices()[res * (res - 3) + (i + 1) % res],
	//			nullptr, nullptr, nullptr,
	//			vNBottom,
	//			&mesh->GetMeshStructure()->GetNormals()[res * (res - 3) + i % res],
	//			&mesh->GetMeshStructure()->GetNormals()[res * (res - 3) + (i + 1) % res]);
	//	}

	//	// middle layers
	//	for (uint32_t i = 0; i < res - 3; ++i)
	//	{
	//		for (uint32_t j = 0; j < res; ++j)
	//		{
	//			mesh->GetMeshStructure()->CreateTriangle(
	//				&mesh->GetMeshStructure()->GetVertices()[i * res + j], 
	//				&mesh->GetMeshStructure()->GetVertices()[(i + 1) * res + (j + 1) % res], 
	//				&mesh->GetMeshStructure()->GetVertices()[(i + 1) * res + j],
	//				nullptr, nullptr, nullptr,
	//				&mesh->GetMeshStructure()->GetNormals()[i * res + j],
	//				&mesh->GetMeshStructure()->GetNormals()[(i + 1) * res + (j + 1) % res],
	//				&mesh->GetMeshStructure()->GetNormals()[(i + 1) * res + j]);

	//			mesh->GetMeshStructure()->CreateTriangle(
	//				&mesh->GetMeshStructure()->GetVertices()[i * res + j], 
	//				&mesh->GetMeshStructure()->GetVertices()[i * res + (j + 1) % res], 
	//				&mesh->GetMeshStructure()->GetVertices()[(i + 1) * res + (j + 1) % res],
	//				nullptr, nullptr, nullptr,
	//				&mesh->GetMeshStructure()->GetNormals()[i * res + j],
	//				&mesh->GetMeshStructure()->GetNormals()[i * res + (j + 1) % res],
	//				&mesh->GetMeshStructure()->GetNormals()[(i + 1) * res + (j + 1) % res]);
	//		}
	//	}


	//	// triangle coloring
	//	for (uint32_t i = 0; i < mesh->GetMeshStructure()->GetTriangles().GetCapacity(); ++i)
	//	{
	//		mesh->GetMeshStructure()->GetTriangles()[i].color = 
	//			Graphics::Color(
	//				rand() % 63 + 192,
	//				rand() % 63 + 192, 
	//				rand() % 63 + 192,
	//				0x00);
	//	}
	//}
}