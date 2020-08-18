#include "cuda_engine_kernel.cuh"
#include "cuda_render_parts.cuh"

namespace RayZath
{
	namespace CudaKernel
	{
		__global__ void GenerateCameraRay(
			CudaKernelData* const kernel_data,
			CudaWorld* const world,
			const int camera_id)
		{
			CudaCamera* const camera = &world->cameras[camera_id];
			if (!camera->Exist()) return;

			const size_t camera_width = camera->width;
			const size_t camera_height = camera->height;

			// calculate which pixel the thread correspond to
			const size_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
			if (thread_index >= camera_width * camera_height) return;

			if (thread_index == 0)
			{
				camera->rotation /= 2.0f;
				camera->rotation = cudaVec3<float>(0.0f, 0.01f, 0.01f);
			}
			__syncthreads();


			const size_t thread_x = thread_index % camera_width;
			const size_t thread_y = thread_index / camera_width;


			RayIntersection intersection;
			intersection.worldSpaceRay.direction = cudaVec3<float>(0.0f, 0.0f, 1.0f);

			// ray to screen deflection
			float xShift = __tanf(camera->fov * 0.5f);
			float yShift = -__tanf(camera->fov * 0.5f) / camera->aspect_ratio;
			intersection.worldSpaceRay.direction.x = ((thread_x / (float)camera_width - 0.5f) * xShift);
			intersection.worldSpaceRay.direction.y = ((thread_y / (float)camera_height - 0.5f) * yShift);

			// pixel position distortion (antialiasing)
			intersection.worldSpaceRay.direction.x +=
				((0.5f / (float)camera_width) * kernel_data->randomNumbers.GetSignedUniform());
			intersection.worldSpaceRay.direction.y +=
				((0.5f / (float)camera_height) * kernel_data->randomNumbers.GetSignedUniform());

			// focal point
			cudaVec3<float> focalPoint = intersection.worldSpaceRay.direction * camera->focal_distance;

			// aperture distortion
			float apertureAngle = kernel_data->randomNumbers.GetUnsignedUniform() * 6.28318530f;
			float apertureSample = kernel_data->randomNumbers.GetUnsignedUniform() * camera->aperture;
			intersection.worldSpaceRay.origin += cudaVec3<float>(
				apertureSample * __sinf(apertureAngle),
				apertureSample * __cosf(apertureAngle),
				0.0f);

			// depth of field ray
			intersection.worldSpaceRay.direction = focalPoint - intersection.worldSpaceRay.origin;


			// ray direction rotation
			intersection.worldSpaceRay.direction.RotateZ(camera->rotation.z);
			intersection.worldSpaceRay.direction.RotateX(camera->rotation.x);
			intersection.worldSpaceRay.direction.RotateY(camera->rotation.y);
			intersection.worldSpaceRay.direction.Normalize();

			// ray origin rotation
			intersection.worldSpaceRay.origin.RotateZ(camera->rotation.z);
			intersection.worldSpaceRay.origin.RotateX(camera->rotation.x);
			intersection.worldSpaceRay.origin.RotateY(camera->rotation.y);

			// ray transposition
			intersection.worldSpaceRay.origin += camera->position;


			// trace ray from camera
			TracingPath* tracingPath = &camera->GetTracingPath(thread_index);
			tracingPath->ResetPath();

			//camera->SamplingImagePixel(thread_index) += CudaColor<float>(0.0f, 1.0f, 0.0f);
			//return;

			TraceRay(*kernel_data, *world, *tracingPath, intersection);
			camera->SamplingImagePixel(thread_index) += tracingPath->CalculateFinalColor();
		}

		__device__ void TraceRay(
			CudaKernelData& kernel_data,
			const CudaWorld& world,
			TracingPath& tracing_path,
			RayIntersection& ray_intersection)
		{
			CudaColor<float> colorMask(1.0f, 1.0f, 1.0f);
			CudaColor<float> lightColor;

			do
			{
				// find closest intersection with world objects
				if (!ClosestIntersection(world, ray_intersection))
				{	// no intersection with world objets

					// check intersection with lights
					LightIntersection lightIntersect;
					LightsIntersection(world, ray_intersection.worldSpaceRay, lightIntersect);

					// calculate final color
					tracing_path.finalColor += CudaColor<float>::BlendProduct(
						colorMask,
						CudaColor<float>::BlendAverage(CudaColor<float>(1.0f, 1.0f, 1.0f) * 0.1f,
							lightIntersect.lightColor,
							lightIntersect.blendFactor));

					return;
				}


				// check intersection with lights
				LightIntersection light_intersection;
				LightsIntersection(world, ray_intersection.worldSpaceRay, light_intersection);

				// check if the object is emitting light
				if (ray_intersection.material.emission > 0.0f)
				{
					tracing_path.finalColor += CudaColor<float>::BlendProduct(
						colorMask,
						CudaColor<float>::BlendAverage(ray_intersection.surfaceColor * ray_intersection.material.emission,
							light_intersection.lightColor,
							light_intersection.blendFactor));
					return;
				}

				if (ray_intersection.material.type != MaterialType::Specular)
					lightColor = TraceLightRays(kernel_data, world, ray_intersection);
				else lightColor = CudaColor<float>(0.0f, 0.0f, 0.0f);


				// calculate final color
				colorMask.BlendProduct(ray_intersection.surfaceColor);
				tracing_path.finalColor += CudaColor<float>::BlendProduct(
					colorMask,
					CudaColor<float>::BlendAverage(
						CudaColor<float>::BlendProduct(ray_intersection.surfaceColor, lightColor),
						light_intersection.lightColor,
						light_intersection.blendFactor));

				if (!tracing_path.NextNodeAvailable()) return;

/* delete return */				return;
				//if (ray_intersection.object->material.type != MaterialType::MaterialTypeSpecular) return;

				// generate next ray
				ray_intersection.GenerateNextRay(kernel_data);


			} while (tracing_path.FindNextNodeToTrace());
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
					intersection = currentIntersection;
					closest_object = sphere;
				}
			}


			//// [>] Check every single mesh
			//for (unsigned int index = 0u, tested = 0u; (index < World.meshes.GetCapacity() && tested < World.meshes.GetCount()); ++index)
			//{
			//	if (!World.meshes[index].Exist()) continue;
			//	const CudaMesh* mesh = &World.meshes[index];
			//	++tested;

			//	if (mesh->RayIntersect(currentIntersection))
			//	{
			//		intersection = currentIntersection;
			//		closest_object = mesh;
			//	}
			//}

			if (closest_object)
			{
				intersection.material = closest_object->material;
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


			// [>] Test intersection with every sphere
			for (unsigned int index = 0u, tested = 0u; (index < world.spheres.GetCapacity() && tested < world.spheres.GetCount()); ++index)
			{
				if (!world.spheres[index].Exist()) continue;
				const CudaSphere* sphere = &world.spheres[index];
				++tested;

				if (sphere->ShadowRayIntersect(shadow_ray)) return 0.0f;
			}


			//// [>] test intersection with every mesh
			//for (unsigned int index = 0u, tested = 0u; (index < world.meshes.GetCapacity() && tested < world.meshes.GetCount()); ++index)
			//{
			//	if (!world.meshes[index].Exist()) continue;
			//	const CudaMesh* mesh = &world.meshes[index];
			//	++tested;

			//	if (mesh->ShadowRayIntersect(shadowRay)) return 0.0f;
			//}

			return 1.0f;
		}
		__device__ void LightsIntersection(
			const CudaWorld& world,
			const CudaRay& ray,
			LightIntersection& intersection)
		{
			// [>] PointLights
			for (unsigned int index = 0u, tested = 0u; (index < world.pointLights.GetCapacity() && tested < world.pointLights.GetCount()); ++index)
			{
				const CudaPointLight* pointLight = &world.pointLights[index];
				if (!pointLight->Exist()) continue;
				++tested;

				cudaVec3<float> vPL = pointLight->position - ray.origin;
				if (vPL.Magnitude() >= ray.length) continue;
				if (cudaVec3<float>::DotProduct(vPL, ray.direction) < 0.0f) continue;


				float dist = RayToPointDistance(ray, pointLight->position);
				if (dist < pointLight->size)
				{
					intersection.lightColor = pointLight->color * pointLight->emission;
					intersection.blendFactor = 0.0f;
					return;
				}
			}


			//// [>] SpotLights
			//for (unsigned int index = 0u, tested = 0u; (index < world.spotLights.GetCapacity() && tested < world.spotLights.GetCount()); ++index)
			//{
			//	CudaSpotLight* spotLight = &world.spotLights[index];
			//	if (spotLight->DoesNotExist()) continue;
			//	++tested;

			//	cudaVec3<float> vPL = spotLight->position - ray.origin;
			//	if (vPL.Magnitude() >= ray.length) continue;
			//	if (cudaVec3<float>::DotProduct(vPL, ray.direction) < 0.0f) continue;


			//	float dist = RayToPointDistance(ray, spotLight->position);
			//	if (dist < spotLight->size)
			//	{
			//		float beamIllum = 1.0f;
			//		float LP_dot_D = cudaVec3<float>::Similarity(-vPL, spotLight->direction);
			//		if (LP_dot_D < spotLight->cosAngleMax) beamIllum = 0.0f;
			//		else if (LP_dot_D > spotLight->cosAngleMin) beamIllum = 1.0f;
			//		else beamIllum = (LP_dot_D - spotLight->cosAngleMax) / (spotLight->cosAngleMin - spotLight->cosAngleMax);

			//		if (beamIllum > 0.0f)
			//		{
			//			intersection.lightColor = spotLight->color * spotLight->emission * beamIllum;
			//			intersection.blendFactor = 0.0f;
			//			return;
			//		}
			//	}
			//}


			//// [>] DirectLights
			//if (!(ray.length < 3.402823466e+38f))
			//{
			//	for (unsigned int index = 0u, tested = 0u; (index < world.directLights.GetCapacity() && tested < world.directLights.GetCount()); ++index)
			//	{
			//		CudaDirectLight* directLight = &world.directLights[index];
			//		if (directLight->DoesNotExist()) continue;
			//		++tested;

			//		float dot = cudaVec3<float>::Similarity(ray.direction, -directLight->direction);
			//		if (dot > directLight->cosMaxAngularSize)
			//		{
			//			intersection.lightColor = directLight->color * directLight->emission * 100.0f;
			//			intersection.blendFactor = 0.0f;
			//			return;
			//		}
			//	}
			//}
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
			float PLdist = 0.0f;

			CudaColor<float> accLightColor(0.0f, 0.0f, 0.0f);

			// [>] PointLights
			for (unsigned int index = 0u, tested = 0u; (index < world.pointLights.GetCapacity() && tested < world.pointLights.GetCount()); ++index)
			{
				const CudaPointLight* pointLight = &world.pointLights[index];
				if (!pointLight->Exist()) continue;
				++tested;


				// randomize point light position
				cudaVec3<float> distLightPos = pointLight->position + cudaVec3<float>(
					kernel_data.randomNumbers.GetSignedUniform(),
					kernel_data.randomNumbers.GetSignedUniform(),
					kernel_data.randomNumbers.GetSignedUniform()) * pointLight->size;

				// vector from point to light position
				vPL = distLightPos - intersection.worldPoint;

				// dot product with surface normal
				vPL_dot_vN = cudaVec3<float>::Similarity(vPL, intersection.worldNormal);
				if (vPL_dot_vN <= 0.0f) continue;

				// calculate light energy P
				PLdist = vPL.Magnitude();
				distFactor = 1.0f / (PLdist * PLdist + 1.0f);
				float energyAtP = pointLight->emission * distFactor * vPL_dot_vN;
				if (energyAtP < 0.001f) continue;	// unimportant light contribution

				// cast shadow ray and calculate color contribution
				CudaRay shadowRay(intersection.worldPoint + intersection.worldNormal * 0.001f, vPL, PLdist);
				accLightColor += pointLight->color * energyAtP * AnyIntersection(kernel_data, world, shadowRay);

			}


			//// [>] SpotLights
			//for (unsigned int index = 0u, tested = 0u; (index < world.spotLights.GetCapacity() && tested < world.spotLights.GetCount()); ++index)
			//{
			//	CudaSpotLight* spotLight = &world.spotLights[index];
			//	if (spotLight->DoesNotExist()) continue;
			//	++tested;

			//	// randomize spot light position
			//	cudaVec3<float> distLightPos = spotLight->position + cudaVec3<float>(
			//		renderingKernel.randomNumbers.GetSignedUniform(),
			//		renderingKernel.randomNumbers.GetSignedUniform(),
			//		renderingKernel.randomNumbers.GetSignedUniform()) * spotLight->size;

			//	// vector from point to light position
			//	vPL = distLightPos - intersection.worldPoint;

			//	// dot product with surface normal
			//	vPL_dot_vN = cudaVec3<float>::Similarity(vPL, intersection.worldNormal);
			//	if (vPL_dot_vN <= 0.0f) continue;

			//	// calculate light energy at P
			//	PLdist = vPL.Magnitude();
			//	distFactor = 1.0f / (PLdist * PLdist + 1.0f);

			//	float beamIllum = 1.0f;
			//	float LP_dot_D = cudaVec3<float>::Similarity(-vPL, spotLight->direction);
			//	if (LP_dot_D < spotLight->cosAngleMax) beamIllum = 0.0f;
			//	else if (LP_dot_D > spotLight->cosAngleMin) beamIllum = 1.0f;
			//	else beamIllum = (LP_dot_D - spotLight->cosAngleMax) / (spotLight->cosAngleMin - spotLight->cosAngleMax);

			//	float energyAtP = spotLight->emission * distFactor * beamIllum * vPL_dot_vN;
			//	if (energyAtP < 0.001f) continue;	// unimportant light contribution

			//	// cast shadow ray and calculate color contribution
			//	CudaRay shadowRay(intersection.worldPoint + intersection.worldNormal * 0.001f, vPL, PLdist);
			//	accLightColor += spotLight->color * energyAtP * AnyIntersection(world, renderingKernel, shadowRay);
			//}


			//// [>] DirectLights
			//for (unsigned int index = 0u, tested = 0u; (index < world.directLights.GetCapacity() && tested < world.directLights.GetCount()); ++index)
			//{
			//	CudaDirectLight* directLight = &world.directLights[index];
			//	if (directLight->DoesNotExist()) continue;
			//	++tested;

			//	//// vector from point to direct light (reversed direction)
			//	//vPL = -directLight->direction;

			//	// vector from point to direct light (reversed direction)
			//	CudaEngineKernel::RandomVectorOnAngularSphere(renderingKernel.randomNumbers.GetUnsignedUniform(),
			//												  renderingKernel.randomNumbers.GetUnsignedUniform() * directLight->maxAngularSize,
			//												  -directLight->direction, vPL);

			//	// dot product with sufrace normal
			//	vPL_dot_vN = cudaVec3<float>::Similarity(vPL, intersection.worldNormal);
			//	if (vPL_dot_vN <= 0.0f) continue;

			//	// calculate light energy at P
			//	float energyAtP = directLight->emission * vPL_dot_vN;
			//	if (energyAtP < 0.001f) continue;	// unimportant light contribution

			//	// cast shadow ray and calculate color contribution
			//	CudaRay shadowRay(intersection.worldPoint + intersection.worldNormal * 0.001f, vPL);
			//	accLightColor += directLight->color * energyAtP * AnyIntersection(world, renderingKernel, shadowRay);
			//}

			return accLightColor;
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