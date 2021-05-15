# RayZath
RayZath is a 3D Monte Carlo path tracing renderer written in C++ and CUDA. It's a learning project involving topics like: graphics rendering, radiosity, light transport, rendering equation, linear algebra, GPGPU programming, micro optimization, asynchronous execution and efficient memory management.

|  |  |
| :----: | :----: |
| ![teapot normal per vertex 7k spp](https://user-images.githubusercontent.com/38960244/118351955-13953c00-b55f-11eb-94f4-6a9153c22eed.jpg) | ![environment 4 5k](https://user-images.githubusercontent.com/38960244/118352041-8a323980-b55f-11eb-81c9-41e2869a40a2.jpg) |
| ![mirror room 16k spp](https://user-images.githubusercontent.com/38960244/118352323-214bc100-b561-11eb-9fec-6948cf50644d.jpg) | ![living rom GI 14k](https://user-images.githubusercontent.com/38960244/118352120-02006400-b560-11eb-8919-dbf7df42c963.jpg) |

TODO: Full Gellery link

## Features
- Renderable objects: Spheres, Planes, Meshes
- Lights: Direct lights, Point lights, Spot lights
- Materials: Mirror, Glossy, Diffuse, Refractive, Scattering, Transparent, Emissive
- Camera: Field of view, Depth of view, Aperture, Exposure time, Cumulative sampling, Progressive rendering
- Textures
- Acceleration structure: oct-tree bounding volume hierarchy
- Asynchronous GPU/CPU rendering
- System of handles and smart pointers
- Resource sharing and mesh instantiation
