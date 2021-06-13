# RayZath
RayZath is a 3D Monte Carlo path tracing renderer written in C++ and CUDA. It's a learning project involving topics like: graphics rendering, radiosity, light transport, rendering equation, linear algebra, GPGPU programming, micro optimization, asynchronous execution and efficient memory management.

|  |  |
| :----: | :----: |
| ![teapot normal per vertex 7k spp](https://user-images.githubusercontent.com/38960244/118351955-13953c00-b55f-11eb-94f4-6a9153c22eed.jpg) | ![environment 4 5k](https://user-images.githubusercontent.com/38960244/118352041-8a323980-b55f-11eb-81c9-41e2869a40a2.jpg) |
| ![mirror room 16k spp](https://user-images.githubusercontent.com/38960244/118352323-214bc100-b561-11eb-9fec-6948cf50644d.jpg) | ![living rom GI 14k](https://user-images.githubusercontent.com/38960244/118352120-02006400-b560-11eb-8919-dbf7df42c963.jpg) |

For more renders and progress click [here](GalleryOfProgress).

## Features
- Renderable objects: Spheres, Planes, Meshes
- Lights: Direct lights, Point lights, Spot lights
- Materials: Mirror, Glossy, Diffuse, Refractive, Scattering, Transparent, Emissive
- Camera: Field of view, Depth of view, Aperture, Exposure time, Cumulative sampling, Progressive rendering
- Mapping: textures, normals, metalness, specularity, roughness, emission
- Acceleration structure: oct-tree bounding volume hierarchy
- Asynchronous GPU/CPU rendering
- System of handles and smart pointers
- Resource sharing and mesh instantiation

## Usage
- include RayZath to project
  ```
  #include "rayzath.h"
  namespace RZ = RayZath;
  ```
- get engine and world
  ```
  auto& engine = RZ::Engine::GetInstance();
  auto& world = engine.GetWorld();
  ```
- create camera
  ```
  auto camera = world.Container<RZ::World::ContainerType::Camera>().Create(
		RZ::ConStruct<RZ::Camera>(
			L"camera",
			Math::vec3f(0.0f, 1.5f, -5.5f),
			Math::vec3f(0.0f, 0.0f, 0.0f),
			1280u, 720u,
			Math::angle_degf(100.0f),
			5.5f, 0.02f, 0.016f, true));
  ```
- create point light
  ```
  auto point_light = world.Container<RZ::World::ContainerType::PointLight>().Create(
		RZ::ConStruct<RZ::PointLight>(
			L"point light",
			Math::vec3f(2.0f, 3.0f, -2.0f),
			Graphics::Color::Palette::White,
			0.1f, 50.0f));
  ```
- create sphere
  ```
  auto sphere = world.Container<RZ::World::ContainerType::Sphere>().Create(
		RZ::ConStruct<RZ::Sphere>(
			L"sphere",
			Math::vec3f(0.0f, 1.0f, 0.0f),
			Math::vec3f(0.0f, 0.0f, 0.0f),
			Math::vec3f(0.0f, 0.0f, 0.0f),
			Math::vec3f(1.0f, 1.0f, 1.0f),
			world.GenerateMaterial<RZ::Material::Common::Paper>(),
			0.5f));
  ```
- render world
  ```
  engine.RenderWorld();
  ```
- get rendered image
  ```
  Graphics::Bitmap image = camera->GetImageBuffer();
  ```
  Now image contains render after first pass. You can display it wherever you want, or you can use GraphicsBox control to draw it using my small Win32 framework available [here](https://github.com/Greketrotny/WinApiFramework).
  
  
## Dependences
RayZath uses my other two small libraries: 
- [Graphics](https://github.com/Greketrotny/Graphics)
- [Math](https://github.com/Greketrotny/Math)
