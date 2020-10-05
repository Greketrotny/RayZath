#include "cuda_engine_kernel.cuh"

namespace RayZath
{
	namespace CudaKernel
	{
		__global__ void GenerateCameraRay(
			CudaKernelData* const kernel,
			CudaWorld* world,
			const int camera_id)
		{
			CudaCamera* const camera = &world->cameras[camera_id];

			// calculate thread position
			const size_t thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			const size_t thread_y = blockIdx.y * blockDim.y + threadIdx.y;
			if (thread_x >= camera->width || thread_y >= camera->height) return;


			// create intersection object
			RayIntersection intersection;
			intersection.ray.direction = cudaVec3<float>(0.0f, 0.0f, 1.0f);

			// ray to screen deflection
			float xShift = __tanf(camera->fov * 0.5f);
			float yShift = -xShift / camera->aspect_ratio;
			intersection.ray.direction.x = ((thread_x / (float)camera->width - 0.5f) * xShift);
			intersection.ray.direction.y = ((thread_y / (float)camera->height - 0.5f) * yShift);

			// pixel position distortion (antialiasing)
			intersection.ray.direction.x +=
				((0.5f / (float)camera->width) * kernel->randomNumbers.GetSignedUniform());
			intersection.ray.direction.y +=
				((0.5f / (float)camera->height) * kernel->randomNumbers.GetSignedUniform());

			// focal point
			cudaVec3<float> focalPoint = intersection.ray.direction * camera->focal_distance;

			// aperture distortion
			float apertureAngle = kernel->randomNumbers.GetUnsignedUniform() * 6.28318530f;
			float apertureSample = sqrtf(kernel->randomNumbers.GetUnsignedUniform()) * camera->aperture;
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
			TracingPath* tracingPath = &camera->GetTracingPath(thread_y * camera->width + thread_x);
			tracingPath->ResetPath();

			//camera->AppendSample(CudaColor<float>(0.0f, 1.0f, 0.0f), thread_x, thread_y);
			//return;

			TraceRay(*kernel, *world, *tracingPath, intersection);
			camera->AppendSample(tracingPath->CalculateFinalColor(), thread_x, thread_y);
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

				color_mask *= intersection.bvh_factor;

				if (!(light_hit || object_hit))
				{	// no hit, return background color

					tracing_path.finalColor += CudaColor<float>::BlendProduct(
						color_mask,
						CudaColor<float>(1.0f, 1.0f, 1.0f) * 2.0f);
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


					// TODO: implement Beer's law. 
					// P0 - light energy in front of an object
					// P - light energy after going through an object
					// A - absorbance

					// e - material absorbance (constant)
					// b - distance traveled in an object
					// c - molar concentration (constant)

					// P = P0 * 10 ^ -(e * b * c)
				}
				else
				{
					color_mask.BlendProduct(intersection.surface_color);
				}



				// [>] Generate next ray
				if (intersection.material.transmitance > 0.0f)
				{	// ray fallen into material/object					

					return;
					//GenerateTransmissiveRay(kernel, intersection);
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
					{	// glossy reflection

						GenerateGlossyRay(kernel, intersection);
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

			// ~~~~ linear search ~~~~
			/*// [>] Check every single sphere
			for (unsigned int index = 0u, tested = 0u; 
				(index < World.spheres.GetContainer().GetCapacity() && 
					tested < World.spheres.GetContainer().GetCount());
				++index)
			{
				if (!World.spheres.GetContainer()[index].Exist()) continue;
				const CudaSphere* sphere = &World.spheres.GetContainer()[index];
				++tested;

				if (sphere->RayIntersect(currentIntersection))
				{
					closest_object = sphere;
				}
			}*/
			World.spheres.GetBVH().Traverse(currentIntersection, closest_object);
			World.meshes.GetBVH().Traverse(currentIntersection, closest_object);

			// copy intersection
			intersection.bvh_factor = currentIntersection.bvh_factor;
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
			for (unsigned int index = 0u, tested = 0u; 
				(index < world.spheres.GetContainer().GetCapacity() && 
					tested < world.spheres.GetContainer().GetCount());
				++index)
			{
				if (!world.spheres.GetContainer()[index].Exist()) continue;
				const CudaSphere* sphere = &world.spheres.GetContainer()[index];
				++tested;

				total_shadow *= sphere->ShadowRayIntersect(shadow_ray);
				if (total_shadow < 0.0001f) return total_shadow;
			}


			// [>] test intersection with every mesh
			for (unsigned int index = 0u, tested = 0u; 
				(index < world.meshes.GetContainer().GetCapacity() && 
					tested < world.meshes.GetContainer().GetCount()); 
				++index)
			{
				if (!world.meshes.GetContainer()[index].Exist()) continue;
				const CudaMesh* mesh = &world.meshes.GetContainer()[index];
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
				vPL_dot_vN = cudaVec3<float>::Similarity(vPL, intersection.mapped_normal);
				if (vPL_dot_vN <= 0.0f) continue;

				// calculate light energy P
				dPL = vPL.Magnitude();
				distFactor = 1.0f / (dPL * dPL + 1.0f);
				float energyAtP = point_light->emission * distFactor * vPL_dot_vN;
				if (energyAtP < 0.001f) continue;	// unimportant light contribution

				// cast shadow ray and calculate color contribution
				CudaRay shadowRay(intersection.point + intersection.surface_normal * 0.0001f, vPL, dPL);
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
				vPL_dot_vN = cudaVec3<float>::Similarity(vPL, intersection.mapped_normal);
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
				CudaRay shadowRay(intersection.point + intersection.surface_normal * 0.001f, vPL, dPL);
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
				vPL_dot_vN = cudaVec3<float>::Similarity(vPL, intersection.mapped_normal);
				if (vPL_dot_vN <= 0.0f) continue;

				// calculate light energy at P
				float energyAtP = directLight->emission * vPL_dot_vN;
				if (energyAtP < 0.001f) continue;	// unimportant light contribution

				// cast shadow ray and calculate color contribution
				CudaRay shadowRay(intersection.point + intersection.surface_normal * 0.0001f, vPL);
				accLightColor += directLight->color * energyAtP * AnyIntersection(kernel_data, world, shadowRay);
			}

			return accLightColor;
		}

		__device__ void GenerateDiffuseRay(
			CudaKernelData& kernel,
			RayIntersection& intersection)
		{
			cudaVec3<float> sample = CosineSampleHemisphere(
				kernel.randomNumbers.GetUnsignedUniform(),
				kernel.randomNumbers.GetUnsignedUniform(),
				intersection.mapped_normal);
			sample.Normalize();

			// flip sample above surface if needed
			const float vR_dot_vN = cudaVec3<float>::Similarity(sample, intersection.surface_normal);
			if (vR_dot_vN < 0.0f) sample += intersection.surface_normal * -2.0f * vR_dot_vN;

			new (&intersection.ray) CudaSceneRay(
				intersection.point + intersection.surface_normal * 0.0001f,
				sample,
				intersection.ray.material);
		}
		__device__ void GenerateSpecularRay(
			CudaKernelData& kernel,
			RayIntersection& intersection)
		{
			cudaVec3<float> reflect = ReflectVector(
				intersection.ray.direction,
				intersection.mapped_normal);

			// flip sample above surface if needed
			const float vR_dot_vN = cudaVec3<float>::Similarity(reflect, intersection.surface_normal);
			if (vR_dot_vN < 0.0f) reflect += intersection.surface_normal * -2.0f * vR_dot_vN;

			new (&intersection.ray) CudaSceneRay(
				intersection.point + intersection.surface_normal * 0.0001f,
				reflect, intersection.ray.material);
		}
		__device__ void GenerateGlossyRay(
			CudaKernelData& kernel,
			RayIntersection& intersection)
		{
			if (intersection.material.glossiness > 0.0f)
			{
				const cudaVec3<float> vNd = SampleHemisphere(
					kernel.randomNumbers.GetUnsignedUniform(),
					1.0f - __powf(
						kernel.randomNumbers.GetUnsignedUniform(),
						intersection.material.glossiness),
					intersection.mapped_normal);

				// calculate reflection direction
				cudaVec3<float> vR = ReflectVector(
					intersection.ray.direction,
					vNd);

				// reflect sample above surface if needed
				const float vR_dot_vN = cudaVec3<float>::Similarity(vR, intersection.surface_normal);
				if (vR_dot_vN < 0.0f) vR += intersection.surface_normal * -2.0f * vR_dot_vN;

				// create next glossy CudaSceneRay
				new (&intersection.ray) CudaSceneRay(
					intersection.point + intersection.surface_normal * 0.0001f,
					vR,
					intersection.ray.material);
			}
			else
			{	// minimum/zero glossiness = perfect mirror

				GenerateSpecularRay(kernel, intersection);
			}

			/*
			* GlossySpecular::sample_f(const ShadeRec& sr,
				const Vector3D& wo,
				Vector3D& wi,
				float& pdf) const
			{
				float ndotwo = sr.normal * wo;
				Vector3D r = -wo + 2.0 * sr.normal * ndotwo; // direction of mirror reflection


				Vector3D w = r;
				Vector3D u = Vector3D(0.00424, 1, 0.00764) ^ w;
				u.normalize();
				Vector3D v = u ^ w;

				Point3D sp = sampler_ptr->sample_hemisphere();
				wi = sp.x * u + sp.y * v + sp.z * w; // reflected ray direction

				if (sr.normal * wi < 0.0) // reflected ray is below surface
				wi = -sp.x * u - sp.y * v + sp.z * w;

				float phong_lobe = pow(r * wi, exp);
				pdf = phong_lobe * (sr.normal * wi);

				return (ks * cs * phong_lobe);
			}
			*/
		}
		//__device__ void GenerateTransmissiveRay(
		//	CudaKernelData& kernel,
		//	RayIntersection& intersection)
		//{
		//	if (intersection.material.ior > 1.0f || intersection.ray.material.ior > 1.0f)
		//	{	// refraction ray

		//		const float cosi = fabsf(cudaVec3<float>::DotProduct(
		//			intersection.ray.direction, intersection.mapped_normal));

		//		// calculate sin^2 theta from Snell's law
		//		const float n1 = intersection.ray.material.ior;
		//		const float n2 = intersection.material.ior;
		//		const float ratio = n1 / n2;
		//		const float sin2_t = ratio * ratio * (1.0f - cosi * cosi);

		//		if (sin2_t >= 1.0f)
		//		{	// TIR

		//			// calculate reflection vector
		//			const cudaVec3<float> reflect = ReflectVector(
		//				intersection.ray.direction,
		//				intersection.normal);

		//			// create new internal reflection CudaSceneRay
		//			new (&intersection.ray) CudaSceneRay(
		//				intersection.point + intersection.normal * 0.0001f,
		//				reflect,
		//				intersection.ray.material);
		//		}
		//		else
		//		{
		//			// calculate fresnel
		//			const float cost = sqrtf(1.0f - sin2_t);
		//			const float Rp = ((n1 * cosi) - (n2 * cost)) / ((n1 * cosi) + (n2 * cost));
		//			const float Rs = ((n2 * cosi) - (n1 * cost)) / ((n2 * cosi) + (n1 * cost));
		//			const float f = (Rs * Rs + Rp * Rp) / 2;

		//			if (f < kernel.randomNumbers.GetUnsignedUniform())
		//			{	// transmission/refraction

		//				// calculate refraction direction
		//				const cudaVec3<float> vR = intersection.ray.direction * ratio +
		//					intersection.normal * (ratio * cosi - cost);

		//				// create new refraction CudaSceneRay
		//				new (&intersection.ray) CudaSceneRay(
		//					intersection.point - intersection.normal * 0.0001f,
		//					vR,
		//					intersection.material);
		//			}
		//			else
		//			{	// reflection

		//				// calculate reflection direction
		//				const cudaVec3<float> reflect = ReflectVector(
		//					intersection.ray.direction,
		//					intersection.normal);

		//				// create new reflection CudaSceneRay
		//				new (&intersection.ray) CudaSceneRay(
		//					intersection.point + intersection.normal * 0.0001f,
		//					reflect,
		//					intersection.ray.material);
		//			}
		//		}
		//	}
		//	else
		//	{	// transparent ray

		//		cudaVec3<float> vD;

		//		if (intersection.material.glossiness > 0.0f)
		//		{
		//			vD = SampleSphere(
		//				kernel.randomNumbers.GetUnsignedUniform(),
		//				1.0f - __powf(
		//					kernel.randomNumbers.GetUnsignedUniform(),
		//					intersection.material.glossiness),
		//				intersection.ray.direction);

		//			const float vS_dot_vN = cudaVec3<float>::Similarity(vD, -intersection.normal);
		//			if (vS_dot_vN < 0.0f) vD += -intersection.normal * -2.0f * vS_dot_vN;
		//		}
		//		else
		//		{
		//			vD = intersection.ray.direction;
		//		}

		//		new (&intersection.ray) CudaSceneRay(
		//			intersection.point - intersection.normal * 0.0001f,
		//			vD,
		//			intersection.material);
		//	}
		//}



		// [>] Tone mapping
		__global__ void ToneMap(
			CudaKernelData* const kernel_data,
			CudaWorld* const world,
			const int camera_id)
		{
			CudaCamera* const camera = &world->cameras[camera_id];

			// calculate thread position
			const size_t thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			const size_t thread_y = blockIdx.y * blockDim.y + threadIdx.y;
			if (thread_x >= camera->width || thread_y >= camera->height) return;

			// average sample color by dividing by number of samples
			CudaColor<float> samplingColor =
				camera->GetSample(thread_x, thread_y) / (float)camera->samples_count;

			// tone map sample color
			camera->SetFinalPixel(kernel_data->renderIndex,
				CudaColor<unsigned char>(
					(samplingColor.red / (samplingColor.red + 1.0f)) * 255.0f,
					(samplingColor.green / (samplingColor.green + 1.0f)) * 255.0f,
					(samplingColor.blue / (samplingColor.blue + 1.0f)) * 255.0f,
					255u),
				thread_x, thread_y);
		}


		// [>] CudaCamera progressive rendering management
		__global__ void CudaCameraSampleReset(
			CudaWorld* const world,
			const int camera_id)
		{
			CudaCamera* const camera = &world->cameras[camera_id];
			if (!camera->Exist()) return;

			// calculate thread position
			const size_t thread_x = blockIdx.x * blockDim.x + threadIdx.x;
			const size_t thread_y = blockIdx.y * blockDim.y + threadIdx.y;
			if (thread_x >= camera->width || thread_y >= camera->height) return;

			// reset sample buffer 
			camera->SetSample(CudaColor<float>(0.0f, 0.0f, 0.0f), thread_x, thread_y);

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




		// shared memory stuff for CudaKernelData and random numbers
		/*
		* 	// copy kernel to shared memory
			extern __shared__ CudaKernelData shared_kernel[];
			CudaKernelData* kernel = shared_kernel;
			const size_t thread_in_block = threadIdx.y * blockDim.x + threadIdx.x;
			const size_t block_in_grid = blockIdx.y * gridDim.x + blockIdx.x;

			if (thread_in_block * block_in_grid == 0)
			{
				kernel->renderIndex = p_kernel->renderIndex;
				kernel->randomNumbers.m_seed = p_kernel->randomNumbers.m_seed + block_in_grid;
			}

			const size_t linear_block_size = blockDim.x * blockDim.y;
			for (int i = 0; i < RandomNumbers::s_count; i += linear_block_size)
			{
				kernel->randomNumbers.m_unsigned_uniform[i + thread_in_block] =
					p_kernel->randomNumbers.m_unsigned_uniform[i + thread_in_block];
				kernel->randomNumbers.m_signed_uniform[i + thread_in_block] =
					p_kernel->randomNumbers.m_signed_uniform[i + thread_in_block];
			}
			__syncthreads();
		*/
	}
}