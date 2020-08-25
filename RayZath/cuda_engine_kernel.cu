#include "cuda_engine_kernel.cuh"
#include "cuda_render_parts.cuh"

namespace RayZath
{
	namespace CudaKernel
	{
		__global__ void GenerateCameraRay(
			CudaKernelData* const kernel_data,
			CudaWorld* world,
			const int camera_id)
		{
			CudaCamera* const camera = &world->cameras[camera_id];
			if (!camera->Exist()) return;

			const size_t camera_width = camera->width;
			const size_t camera_height = camera->height;

			// calculate which pixel the thread correspond to
			const size_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_index >= camera_width * camera_height) return;

			const size_t thread_x = thread_index % camera_width;
			const size_t thread_y = thread_index / camera_width;


			RayIntersection intersection;
			intersection.ray.direction = cudaVec3<float>(0.0f, 0.0f, 1.0f);

			// ray to screen deflection
			float xShift = __tanf(camera->fov * 0.5f);
			float yShift = -__tanf(camera->fov * 0.5f) / camera->aspect_ratio;
			intersection.ray.direction.x = ((thread_x / (float)camera_width - 0.5f) * xShift);
			intersection.ray.direction.y = ((thread_y / (float)camera_height - 0.5f) * yShift);

			// pixel position distortion (antialiasing)
			intersection.ray.direction.x +=
				((0.5f / (float)camera_width) * kernel_data->randomNumbers.GetSignedUniform());
			intersection.ray.direction.y +=
				((0.5f / (float)camera_height) * kernel_data->randomNumbers.GetSignedUniform());

			// focal point
			cudaVec3<float> focalPoint = intersection.ray.direction * camera->focal_distance;

			// aperture distortion
			float apertureAngle = kernel_data->randomNumbers.GetUnsignedUniform() * 6.28318530f;
			float apertureSample = kernel_data->randomNumbers.GetUnsignedUniform() * camera->aperture;
			intersection.ray.origin += cudaVec3<float>(
				apertureSample * __sinf(apertureAngle),
				apertureSample * __cosf(apertureAngle),
				0.0f);

			// depth of field ray
			intersection.ray.direction = focalPoint - intersection.ray.origin;


			// ray direction rotation
			intersection.ray.direction.RotateZ(camera->rotation.z);
			intersection.ray.direction.RotateX(camera->rotation.x);
			intersection.ray.direction.RotateY(camera->rotation.y);
			intersection.ray.direction.Normalize();

			// ray origin rotation
			intersection.ray.origin.RotateZ(camera->rotation.z);
			intersection.ray.origin.RotateX(camera->rotation.x);
			intersection.ray.origin.RotateY(camera->rotation.y);

			// ray transposition
			intersection.ray.origin += camera->position;


			// trace ray from camera
			TracingPath* tracingPath = &camera->GetTracingPath(thread_index);
			tracingPath->ResetPath();

			//camera->SamplingImagePixel(thread_index) += CudaColor<float>(0.0f, 1.0f, 0.0f);
			//return;

			TraceRay(*kernel_data, *world, *tracingPath, intersection);
			camera->SamplingImagePixel(thread_index) += tracingPath->CalculateFinalColor();
		}

		__device__ void TraceRay(
			CudaKernelData& kernel,
			const CudaWorld& world,
			TracingPath& tracing_path,
			RayIntersection& intersection)
		{
			CudaColor<float> color_mask(1.0f, 1.0f, 1.0f);

			do
			{
				bool light_hit = LightsIntersection(world, intersection);
				bool object_hit = ClosestIntersection(world, intersection);

				if (!(light_hit || object_hit))
				{	// no hit, return background color

					tracing_path.finalColor += CudaColor<float>::BlendProduct(
						color_mask,
						CudaColor<float>(1.0f, 1.0f, 1.0f) * 0.1f);
					return;
				}

				if (intersection.material.emitance > 0.0f)
				{	// intersection with emitting object

					tracing_path.finalColor += CudaColor<float>::BlendProduct(
						color_mask,
						intersection.surface_color * intersection.material.emitance);
					return;
				}

				if (!tracing_path.NextNodeAvailable()) return;


				if (intersection.ray.material.transmitance < 1.0f)
				{
					color_mask.BlendProduct(
						intersection.surface_color *
						__powf(intersection.ray.material.transmitance, intersection.ray.length));
				}
				else
				{
					color_mask.BlendProduct(intersection.surface_color);
				}



				// [>] Generate next ray
				if (intersection.material.transmitance > 0.0f)
				{	// ray fallen into material/object					

					GenerateTransmissiveRay(kernel, intersection);
				}
				else
				{	// ray is reflected from sufrace

					if (kernel.randomNumbers.GetUnsignedUniform() > intersection.material.reflectance)
					{	// diffuse reflection

						CudaColor<float> light_color = TraceLightRays(kernel, world, intersection);
						tracing_path.finalColor += CudaColor<float>::BlendProduct(
							color_mask,
							CudaColor<float>::BlendProduct(intersection.surface_color, light_color));

						GenerateDiffuseRay(kernel, intersection);
					}
					else
					{	// specular reflection

						if (!tracing_path.NextNodeAvailable()) return;
						GenerateSpecularRay(kernel, intersection);
					}
				}

			} while (tracing_path.FindNextNodeToTrace());
		}
		__device__ bool LightsIntersection(
			const CudaWorld& world,
			RayIntersection& intersection)
		{
			bool hit = false;

			// [>] PointLights
			for (unsigned int index = 0u, tested = 0u; (index < world.pointLights.GetCapacity() && tested < world.pointLights.GetCount()); ++index)
			{
				const CudaPointLight* pointLight = &world.pointLights[index];
				if (!pointLight->Exist()) continue;
				++tested;

				cudaVec3<float> vPL = pointLight->position - intersection.ray.origin;
				float dPL = vPL.Magnitude();

				// check if light is close enough
				if (dPL >= intersection.ray.length) continue;
				// check if light is in front of ray
				if (cudaVec3<float>::DotProduct(vPL, intersection.ray.direction) < 0.0f) continue;


				float dist = RayToPointDistance(intersection.ray, pointLight->position);
				if (dist < pointLight->size)
				{	// ray intersects with the light
					intersection.ray.length = dPL;
					intersection.surface_color = pointLight->color;
					intersection.material.emitance = pointLight->emission;
					hit = true;
				}
			}


			// [>] SpotLights
			for (unsigned int index = 0u, tested = 0u; (index < world.spotLights.GetCapacity() && tested < world.spotLights.GetCount()); ++index)
			{
				const CudaSpotLight* spotLight = &world.spotLights[index];
				if (!spotLight->Exist()) continue;
				++tested;

				cudaVec3<float> vPL = spotLight->position - intersection.ray.origin;
				float dPL = vPL.Magnitude();

				if (dPL >= intersection.ray.length) continue;
				float vPL_dot_vD = cudaVec3<float>::DotProduct(vPL, intersection.ray.direction);
				if (vPL_dot_vD < 0.0f) continue;

				float dist = RayToPointDistance(intersection.ray, spotLight->position);
				if (dist < spotLight->size)
				{
					float t_dist = sqrtf(
						(spotLight->size + spotLight->sharpness) * 
						(spotLight->size + spotLight->sharpness) - 
						dist * dist);

					cudaVec3<float> test_point =
						intersection.ray.origin + intersection.ray.direction * vPL_dot_vD - 
						intersection.ray.direction * t_dist;

					float LP_dot_D = cudaVec3<float>::Similarity(
						test_point - spotLight->position, spotLight->direction);
					if (LP_dot_D > spotLight->cos_angle)
					{
						intersection.ray.length = dPL;
						intersection.surface_color = spotLight->color;
						intersection.material.emitance = spotLight->emission;
						hit = true;
					}
				}
			}


			// [>] DirectLights
			if (!(intersection.ray.length < 3.402823466e+38f))
			{
				for (unsigned int index = 0u, tested = 0u; (index < world.directLights.GetCapacity() && tested < world.directLights.GetCount()); ++index)
				{
					const CudaDirectLight* directLight = &world.directLights[index];
					if (!directLight->Exist()) continue;
					++tested;

					float dot = cudaVec3<float>::Similarity(intersection.ray.direction, -directLight->direction);
					if (dot > directLight->cos_angular_size)
					{
						intersection.surface_color = directLight->color;
						intersection.material.emitance = directLight->emission;
						hit = true;
					}
				}
			}

			return hit;
		}
		__device__ bool ClosestIntersection(
			const CudaWorld& World,
			RayIntersection& intersection)
		{
			RayIntersection currentIntersection = intersection;
			const CudaRenderObject* closest_object = nullptr;

			// [>] Check every single sphere
			for (unsigned int index = 0u, tested = 0u; (index < World.spheres.GetCapacity() && tested < World.spheres.GetCount()); ++index)
			{
				if (!World.spheres[index].Exist()) continue;
				const CudaSphere* sphere = &World.spheres[index];
				++tested;

				if (sphere->RayIntersect(currentIntersection))
				{
					closest_object = sphere;
				}
			}


			// [>] Check every single mesh
			for (unsigned int index = 0u, tested = 0u; (index < World.meshes.GetCapacity() && tested < World.meshes.GetCount()); ++index)
			{
				if (!World.meshes[index].Exist()) continue;
				const CudaMesh* mesh = &World.meshes[index];
				++tested;

				if (mesh->RayIntersect(currentIntersection))
				{
					closest_object = mesh;
				}
			}

			if (closest_object)
			{
				intersection = currentIntersection;
				return true;
			}
			else return false;
		}
		__device__ float AnyIntersection(
			CudaKernelData& kernel_data,
			const CudaWorld& world,
			const CudaRay& shadow_ray)
		{
			// Legend:
			// L - light position
			// P - point of intersection

			float total_shadow = 1.0f;

			// [>] Test intersection with every sphere
			for (unsigned int index = 0u, tested = 0u; (index < world.spheres.GetCapacity() && tested < world.spheres.GetCount()); ++index)
			{
				if (!world.spheres[index].Exist()) continue;
				const CudaSphere* sphere = &world.spheres[index];
				++tested;

				total_shadow *= sphere->ShadowRayIntersect(shadow_ray);
				if (total_shadow < 0.0001f) return total_shadow;
			}


			// [>] test intersection with every mesh
			for (unsigned int index = 0u, tested = 0u; (index < world.meshes.GetCapacity() && tested < world.meshes.GetCount()); ++index)
			{
				if (!world.meshes[index].Exist()) continue;
				const CudaMesh* mesh = &world.meshes[index];
				++tested;

				total_shadow *= mesh->ShadowRayIntersect(shadow_ray);
				if (total_shadow < 0.0001f) return total_shadow;
			}

			return total_shadow;
		}
		__device__ CudaColor<float> TraceLightRays(
			CudaKernelData& kernel_data,
			const CudaWorld& world,
			RayIntersection& intersection)
		{
			// Legend:
			// L - position of current light
			// P - point of intersetion
			// vN - surface normal

			cudaVec3<float> vPL;

			float distFactor = 1.0f;
			float vPL_dot_vN = 1.0f;
			float dPL = 0.0f;

			CudaColor<float> accLightColor(0.0f, 0.0f, 0.0f);

			// [>] PointLights
			for (unsigned int index = 0u, tested = 0u; (index < world.pointLights.GetCapacity() && tested < world.pointLights.GetCount()); ++index)
			{
				const CudaPointLight* point_light = &world.pointLights[index];
				if (!point_light->Exist()) continue;
				++tested;


				// randomize point light position
				cudaVec3<float> distLightPos = point_light->position + cudaVec3<float>(
					kernel_data.randomNumbers.GetSignedUniform(),
					kernel_data.randomNumbers.GetSignedUniform(),
					kernel_data.randomNumbers.GetSignedUniform()) * point_light->size;

				// vector from point to light position
				vPL = distLightPos - intersection.point;

				// dot product with surface normal
				vPL_dot_vN = cudaVec3<float>::Similarity(vPL, intersection.normal);
				if (vPL_dot_vN <= 0.0f) continue;

				// calculate light energy P
				dPL = vPL.Magnitude();
				distFactor = 1.0f / (dPL * dPL + 1.0f);
				float energyAtP = point_light->emission * distFactor * vPL_dot_vN;
				if (energyAtP < 0.001f) continue;	// unimportant light contribution

				// cast shadow ray and calculate color contribution
				CudaRay shadowRay(intersection.point + intersection.normal * 0.0001f, vPL, dPL);
				accLightColor += point_light->color * energyAtP * AnyIntersection(kernel_data, world, shadowRay);
			}


			// [>] SpotLights
			for (unsigned int index = 0u, tested = 0u; (index < world.spotLights.GetCapacity() && tested < world.spotLights.GetCount()); ++index)
			{
				const CudaSpotLight* spotLight = &world.spotLights[index];
				if (!spotLight->Exist()) continue;
				++tested;

				// randomize spot light position
				cudaVec3<float> distLightPos = spotLight->position + cudaVec3<float>(
					kernel_data.randomNumbers.GetSignedUniform(),
					kernel_data.randomNumbers.GetSignedUniform(),
					kernel_data.randomNumbers.GetSignedUniform()) * spotLight->size;

				// vector from point to light position
				vPL = distLightPos - intersection.point;

				// dot product with surface normal
				vPL_dot_vN = cudaVec3<float>::Similarity(vPL, intersection.normal);
				if (vPL_dot_vN <= 0.0f) continue;

				// calculate light energy at P
				dPL = vPL.Magnitude();
				distFactor = 1.0f / (dPL * dPL + 1.0f);

				float beamIllum = 1.0f;
				float LP_dot_D = cudaVec3<float>::Similarity(-vPL, spotLight->direction);
				if (LP_dot_D < spotLight->cos_angle) beamIllum = 0.0f;
				else beamIllum = 1.0f;

				float energyAtP = spotLight->emission * distFactor * beamIllum * vPL_dot_vN;
				if (energyAtP < 0.001f) continue;	// unimportant light contribution

				// cast shadow ray and calculate color contribution
				CudaRay shadowRay(intersection.point + intersection.normal * 0.001f, vPL, dPL);
				accLightColor += spotLight->color * energyAtP * AnyIntersection(kernel_data, world, shadowRay);
			}


			// [>] DirectLights
			for (unsigned int index = 0u, tested = 0u; (index < world.directLights.GetCapacity() && tested < world.directLights.GetCount()); ++index)
			{
				const CudaDirectLight* directLight = &world.directLights[index];
				if (!directLight->Exist()) continue;
				++tested;

				//// vector from point to direct light (reversed direction)
				//vPL = -directLight->direction;

				// vector from point to direct light (reversed direction)
				RandomVectorOnAngularSphere(
					kernel_data.randomNumbers.GetUnsignedUniform(),
					kernel_data.randomNumbers.GetUnsignedUniform() * directLight->angular_size,
					-directLight->direction, vPL);

				// dot product with sufrace normal
				vPL_dot_vN = cudaVec3<float>::Similarity(vPL, intersection.normal);
				if (vPL_dot_vN <= 0.0f) continue;

				// calculate light energy at P
				float energyAtP = directLight->emission * vPL_dot_vN;
				if (energyAtP < 0.001f) continue;	// unimportant light contribution

				// cast shadow ray and calculate color contribution
				CudaRay shadowRay(intersection.point + intersection.normal * 0.0001f, vPL);
				accLightColor += directLight->color * energyAtP * AnyIntersection(kernel_data, world, shadowRay);
			}

			return accLightColor;
		}

		__device__ void GenerateDiffuseRay(
			CudaKernelData& kernel,
			RayIntersection& intersection)
		{
			cudaVec3<float> sample;
			DirectionOnHemisphere(
				kernel.randomNumbers.GetUnsignedUniform(),
				kernel.randomNumbers.GetUnsignedUniform(),
				intersection.normal, sample);

			new (&intersection.ray) CudaSceneRay(
				intersection.point + intersection.normal * 0.0001f,
				sample,
				intersection.ray.material);
		}
		__device__ void GenerateSpecularRay(
			CudaKernelData& kernel,
			RayIntersection& intersection)
		{
			cudaVec3<float> reflect = ReflectVector(
					intersection.ray.direction,
					intersection.normal);

			new (&intersection.ray) CudaSceneRay(
				intersection.point + intersection.normal * 0.0001f, 
				reflect, intersection.ray.material);
		}
		__device__ void GenerateTransmissiveRay(
			CudaKernelData& kernel,
			RayIntersection& intersection)
		{
			if (intersection.material.ior > 1.0f)
			{	// refraction ray

				cudaVec3<float> vRefract;
				float k = Refract(
					intersection.ray.direction,
					intersection.normal,
					intersection.ray.material.ior,
					intersection.material.ior,
					vRefract);
				if (k < kernel.randomNumbers.GetSignedUniform())
				{	// transmission/refraction

					new (&intersection.ray) CudaSceneRay(
						intersection.point - intersection.normal * 0.0001f,
						vRefract,
						intersection.material);
				}
				else
				{	// TIR (total internal reflection)

					cudaVec3<float> reflect = ReflectVector(
						intersection.ray.direction,
						intersection.normal);

					new (&intersection.ray) CudaSceneRay(
						intersection.point + intersection.normal * 0.0001f,
						reflect, intersection.ray.material);
				}
			}
			else
			{	// transparent ray
				new (&intersection.ray) CudaSceneRay(
					intersection.point - intersection.normal * 0.0001f,
					intersection.ray.direction,
					intersection.material);
			}
		}


		// [>] Tone mapping
		__global__ void ToneMap(
			CudaKernelData* const kernel_data,
			CudaWorld* const world,
			const int camera_id)
		{
			CudaCamera* const camera = &world->cameras[camera_id];

			const unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
			if (threadIndex >= camera->width * camera->height) return;

			// average sample color by dividing by number of samples
			CudaColor<float> samplingColor = camera->SamplingImagePixel(threadIndex) / (float)camera->samples_count;

			// tone map sample color
			camera->FinalImagePixel(kernel_data->renderIndex, threadIndex) =
				CudaColor<unsigned char>(
					(samplingColor.red / (samplingColor.red + 1.0f)) * 255.0f,
					(samplingColor.green / (samplingColor.green + 1.0f)) * 255.0f,
					(samplingColor.blue / (samplingColor.blue + 1.0f)) * 255.0f,
					255u);
		}


		// [>] CudaCamera progressive rendering management
		__global__ void CudaCameraSampleReset(
			CudaWorld* const world, 
			const int camera_id)
		{
			CudaCamera* const camera = &world->cameras[camera_id];
			if (!camera->Exist()) return;

			const unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
			if (threadIndex >= camera->width * camera->height) return;

			// reset sample buffer 
			camera->SamplingImagePixel(threadIndex) = CudaColor<float>(0.0f, 0.0f, 0.0f);

			// TODO: reset tracing paths
		}
		__global__ void CudaCameraUpdateSamplesNumber(
			CudaWorld* const world,
			const int camera_id,
			bool reset_flag)
		{
			CudaCamera* const camera = &world->cameras[camera_id];
			if (reset_flag)	camera->samples_count = 1u;
			else			camera->samples_count += 1u;
		}
	}
}