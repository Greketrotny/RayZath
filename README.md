# RayZath
RayZath is a 3D Monte Carlo path tracing renderer written in C++ and CUDA. It's a learning project involving topics like: graphics rendering, radiosity, light transport, rendering equation, linear algebra, GPGPU programming, micro optimization, asynchronous execution and efficient memory management.

![living_room_ui](https://user-images.githubusercontent.com/38960244/189484419-8464b0e3-8b51-421d-aebf-c5c404db816e.png)
|  |  |
| :----: | :----: |
| ![bugatti](https://user-images.githubusercontent.com/38960244/187255857-1780ba30-ebf0-42b4-88df-7355838c738c.png) | ![teapot normal per vertex 7k spp](https://user-images.githubusercontent.com/38960244/118351955-13953c00-b55f-11eb-94f4-6a9153c22eed.jpg) |
| ![multiple lights 13k](https://user-images.githubusercontent.com/38960244/189485365-3e74a87e-7141-48b4-9c3c-f94d0139dc47.png) | ![billiard](https://user-images.githubusercontent.com/38960244/148647335-e22f5dba-9fbb-4ce1-86ac-83f379169de8.png) |



For more renders and progress click [here](GalleryOfProgress).

## Features
### Engine core
- Renderable objects: Triangle based meshes
- Lights: Direct lights, Spot lights
- Materials: Mirror, Glossy, Diffuse, Refractive, Scattering, Transparent, Emissive
- Camera: Field of view, Depth of view, Aperture, Exposure time
- Mapping: textures, normals, metalness, roughness, emission
- Acceleration structure: oct-tree bounding volume hierarchy
- Rendering:
	- asynchronous GPU/CPU rendering
	- cumulative sampling
	- progressive rendering
	- next event estimation (NEE)
	- multiple importance sampling (MIS)
- System of handles and smart pointers
- Resource sharing and mesh instantiation
- Loading/saving .obj, .mtl files and .json scene description

### GUI
- Real time render viewport preview
- Multiple viewports (resizing, panning, zooming, object selection)
- Editing properties of all components
- Live attaching/detaching maps to/from materials
- Mesh instancing manipulation
- Object groupping
- Loading/Saving maps, materials, meshes, models and scenes
- Common mesh generation (cube, uv sphere, plane, cone, cylinder, torus)
- Common material generation (metals, plastics, translusives, ...)

## Libraries and dependences
- CUDA - gpu rendering
- Vulkan + GLFW - backend for GUI
- [ImGui](https://github.com/ocornut/imgui) - immediate user interface
- [json library](https://github.com/nlohmann/json) - loading/saving json files 
- [stb image](https://github.com/nothings/stb) - loading/saving textures and images
- [Math](https://github.com/Greketrotny/Math) - tiny math library
