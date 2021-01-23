#include "cuda_engine_kernel.cuh"

#include "math_constants.h"

namespace RayZath
{
	namespace CudaEngine
	{
		namespace CudaKernel
		{
			// ~~~~~~~~ Memory Management ~~~~~~~~
			__constant__ CudaConstantKernel const_kernel[2];

			__host__ void CopyToConstantMemory(
				const CudaConstantKernel* hCudaConstantKernel,
				const uint32_t& update_idx,
				cudaStream_t& stream)
			{
				CudaErrorCheck(cudaMemcpyToSymbolAsync(
					(const void*)const_kernel, hCudaConstantKernel,
					sizeof(CudaConstantKernel), update_idx * sizeof(CudaConstantKernel),
					cudaMemcpyKind::cudaMemcpyHostToDevice, stream));
				CudaErrorCheck(cudaStreamSynchronize(stream));
			}


			__device__ CudaConstantKernel* ckernel;

			__global__ void GenerateCameraRay(
				CudaGlobalKernel* const global_kernel,
				CudaWorld* world,
				const int camera_id)
			{
				/*// create local thread structure
				ThreadData thread(global_kernel->randomNumbers.GetSeed(threadIdx.y * blockDim.x + threadIdx.x));

				//CudaKernelData* const kernel = global_kernel;

				// [>] Copy kernel to shared memory
				extern __shared__ CudaKernelData shared_kernel[];
				CudaKernelData* kernel = shared_kernel;

				// copy render index
				if (thread.thread_in_kernel == 0u)
					kernel->renderIndex = global_kernel->renderIndex;

				// copy unsigned random floats
				const uint32_t linear_block_size = blockDim.x * blockDim.y;
				for (uint32_t i = thread.thread_in_block; i < RandomNumbers::s_count; i += linear_block_size)
				{
					kernel->randomNumbers.m_unsigned_uniform[i] =
						global_kernel->randomNumbers.m_unsigned_uniform[i];
				}

				__syncthreads();*/

				// get kernels
				CudaGlobalKernel* const kernel = global_kernel;
				ckernel = &const_kernel[kernel->GetRenderIdx()];

				// create thread object
				ThreadData thread;
				thread.SetSeed(kernel->GetSeeds().GetSeed(thread.thread_in_block));

				// get camera and clamp working threads
				CudaCamera* const camera = &world->cameras[camera_id];
				if (thread.thread_x >= camera->GetWidth() || thread.thread_y >= camera->GetHeight()) return;


				// create intersection object
				RayIntersection intersection;
				intersection.ray.material = &world->material;

				// generate camera ray
				camera->GenerateRay(
					intersection.ray,
					thread,
					*ckernel);

				// trace ray from camera
				TracingPath* tracingPath = 
					&camera->GetTracingPath(thread.thread_y * camera->GetWidth() + thread.thread_x);
				tracingPath->ResetPath();

				/*camera->AppendSample(
					CudaColor<float>(
						intersection.ray.direction.x,
						ckernel->GetRndNumbers().GetUnsignedUniform(thread),
						ckernel->GetRndNumbers().GetUnsignedUniform(thread), 1.0f),
					thread.thread_x, thread.thread_y);
				return;*/

				TraceRay(thread, *world, *tracingPath, intersection);
				camera->AppendSample(tracingPath->CalculateFinalColor(), thread.thread_x, thread.thread_y);

				global_kernel->GetSeeds().SetSeed(thread.seed, thread.thread_in_block);
			}

			__device__ void TraceRay(
				ThreadData& thread,
				const CudaWorld& world,
				TracingPath& tracing_path,
				RayIntersection& intersection)
			{
				CudaColor<float> color_mask(1.0f, 1.0f, 1.0f, 1.0f);

				do
				{
					intersection.surface_material = &world.material;

					if (intersection.ray.material->scattering > 0.0f)
					{
						intersection.ray.length =
							(__logf(1.0f / (ckernel->GetRndNumbers().GetUnsignedUniform(thread) + 1.0e-7f))) /
							intersection.ray.material->scattering;
					}


					if (!world.ClosestIntersection(intersection))
					{
						//color_mask *= intersection.bvh_factor;

						if (intersection.ray.material->scattering > 0.0f)
						{
							// scattering point
							intersection.point = 
								intersection.ray.origin + 
								intersection.ray.direction * intersection.ray.length;
							
							// light illumination at scattering point
							tracing_path.finalColor += 
								CudaColor<float>::BlendProduct(
									color_mask,
									PointDirectSampling(thread, world, intersection));

							// generate scatter direction
							const cudaVec3<float> sctr_direction = SampleSphere(
								ckernel->GetRndNumbers().GetUnsignedUniform(thread),
								ckernel->GetRndNumbers().GetUnsignedUniform(thread),
								intersection.ray.direction);

							// create scattering ray
							new (&intersection.ray) CudaSceneRay(
								intersection.point,
								sctr_direction,
								intersection.ray.material);

							continue;
						}
						else
						{
							tracing_path.finalColor += 
								CudaColor<float>::BlendProduct(
									color_mask,
									intersection.surface_color *
									intersection.surface_material->emittance);
							return;
						}
					}

					//color_mask *= intersection.bvh_factor;


					if (intersection.surface_material->emittance > 0.0f)
					{	// intersection with emitting object

						tracing_path.finalColor += 
							CudaColor<float>::BlendProduct(
								color_mask,
								intersection.surface_color * intersection.surface_material->emittance);
					}


					// [>] apply Beer's law

					// P0 - light energy in front of an object
					// P - light energy after going through an object
					// A - absorbance

					// e - material absorbance (constant)
					// b - distance traveled in an object
					// c - molar concentration (constant)

					// A = 10 ^ -(e * b * c)
					// P = P0 * A

					if (intersection.ray.material->transmittance > 0.0f)
					{
						color_mask.BlendProduct(
							intersection.surface_color *
							__powf(intersection.ray.material->transmittance, intersection.ray.length));
					}



					/*static constexpr float rcp256 = 1.0f / 256.0f;
					static constexpr float max_radiance = 1000.0f;
					static constexpr float min_contribution = rcp256 / max_radiance;
					if (color_mask.red < min_contribution &&
						color_mask.green < min_contribution &&
						color_mask.blue < min_contribution)
						return;*/


					if (!tracing_path.NextNodeAvailable())
						return;


					// [>] Generate next ray
					if (intersection.surface_material->transmittance > 0.0f)
					{	// ray fallen into material/object					

						GenerateTransmissiveRay(thread, intersection);
					}
					else
					{	// ray is reflected from sufrace

						if (ckernel->GetRndNumbers().GetUnsignedUniform(thread) > 
							intersection.surface_material->reflectance)
						{	// diffuse reflection

							tracing_path.finalColor +=
								CudaColor<float>::BlendProduct(
									color_mask,
									CudaColor<float>::BlendProduct(
										intersection.surface_color, 
										SurfaceDirectSampling(thread, world, intersection)));

							GenerateDiffuseRay(thread, intersection);
						}
						else
						{	// glossy reflection

							GenerateGlossyRay(thread, intersection);
						}
					}

				} while (tracing_path.FindNextNodeToTrace());
			}

			__device__ CudaColor<float> SurfaceDirectSampling(
				ThreadData& thread,
				const CudaWorld& world,
				RayIntersection& intersection)
			{
				// Legend:
				// L - position of current light
				// P - point of intersetion
				// vN - surface normal

				CudaColor<float> accLightColor(0.0f, 0.0f, 0.0f, 1.0f);

				// [>] PointLights
				for (uint32_t index = 0u, tested = 0u;
					(index < world.pointLights.GetCapacity() && tested < world.pointLights.GetCount());
					++index)
				{
					const CudaPointLight* point_light = &world.pointLights[index];
					if (!point_light->Exist()) continue;
					++tested;


					// randomize point light position
					const cudaVec3<float> rndL = point_light->position + cudaVec3<float>(
						ckernel->GetRndNumbers().GetUnsignedUniform(thread) * 2.0f - 1.0f,
						ckernel->GetRndNumbers().GetUnsignedUniform(thread) * 2.0f - 1.0f,
						ckernel->GetRndNumbers().GetUnsignedUniform(thread) * 2.0f - 1.0f) * point_light->size;

					// vector from point to light position
					const cudaVec3<float> vPL = rndL - intersection.point;

					// dot product with surface normal
					const float vPL_dot_vN = cudaVec3<float>::Similarity(vPL, intersection.mapped_normal);
					if (vPL_dot_vN <= 0.0f) continue;

					// distance factor (inverse square law)
					const float dPL = vPL.Length();
					const float d_factor = 1.0f / (dPL * dPL + 1.0f);

					// scatering factor
					const float sctr_factor = __expf(-dPL * intersection.ray.material->scattering);

					// calculate radiance at P
					const float radianceP = point_light->material.emittance * d_factor * sctr_factor * vPL_dot_vN;
					if (radianceP < 0.0001f) continue;	// unimportant light contribution

					// cast shadow ray and calculate color contribution
					CudaRay shadowRay(intersection.point + intersection.surface_normal * 0.0001f, vPL, dPL);
					accLightColor += point_light->material.color * radianceP * world.AnyIntersection(shadowRay);
				}


				// [>] SpotLights
				for (uint32_t index = 0u, tested = 0u;
					(index < world.spotLights.GetCapacity() && tested < world.spotLights.GetCount());
					++index)
				{
					const CudaSpotLight* spotLight = &world.spotLights[index];
					if (!spotLight->Exist()) continue;
					++tested;

					// randomize spot light position
					const cudaVec3<float> rndL = spotLight->position + cudaVec3<float>(
						ckernel->GetRndNumbers().GetUnsignedUniform(thread) * 2.0f - 1.0f,
						ckernel->GetRndNumbers().GetUnsignedUniform(thread) * 2.0f - 1.0f,
						ckernel->GetRndNumbers().GetUnsignedUniform(thread) * 2.0f - 1.0f) * spotLight->size;

					// vector from point to light position
					const cudaVec3<float> vPL = rndL - intersection.point;

					// dot product with surface normal
					const float vPL_dot_vN = cudaVec3<float>::Similarity(vPL, intersection.mapped_normal);
					if (vPL_dot_vN <= 0.0f) continue;

					// distance factor (inverse square law)
					const float dPL = vPL.Length();
					const float d_factor = 1.0f / (dPL * dPL + 1.0f);

					// scattering factor
					const float sctr_factor = __expf(-dPL * intersection.ray.material->scattering);

					// beam illumination
					float beamIllum = 1.0f;
					const float LP_dot_D = cudaVec3<float>::Similarity(-vPL, spotLight->direction);
					if (LP_dot_D < spotLight->cos_angle) beamIllum = 0.0f;
					else beamIllum = 1.0f;

					// calculate radiance at P
					const float radianceP = spotLight->material.emittance * d_factor * sctr_factor * beamIllum * vPL_dot_vN;
					if (radianceP < 0.0001f) continue;	// unimportant light contribution

					// cast shadow ray and calculate color contribution
					const CudaRay shadowRay(intersection.point + intersection.surface_normal * 0.001f, vPL, dPL);
					accLightColor += spotLight->material.color * radianceP * world.AnyIntersection(shadowRay);
				}


				// [>] DirectLights
				for (uint32_t index = 0u, tested = 0u;
					(index < world.directLights.GetCapacity() && tested < world.directLights.GetCount());
					++index)
				{
					const CudaDirectLight* directLight = &world.directLights[index];
					if (!directLight->Exist()) continue;
					++tested;

					// vector from point to direct light (reversed direction)
					const cudaVec3<float> vPL = SampleSphere(
						ckernel->GetRndNumbers().GetUnsignedUniform(thread),
						0.5f * (1.0f - __cosf(
							ckernel->GetRndNumbers().GetUnsignedUniform(thread) *
							directLight->angular_size)),
						-directLight->direction);


					// dot product with sufrace normal
					const float vPL_dot_vN = cudaVec3<float>::Similarity(vPL, intersection.mapped_normal);
					if (vPL_dot_vN <= 0.0f) continue;

					// calculate radiance at P
					const float radianceP = directLight->material.emittance * vPL_dot_vN;
					if (radianceP < 0.0001f) continue;	// unimportant light contribution

					// cast shadow ray and calculate color contribution
					CudaRay shadowRay(intersection.point + intersection.surface_normal * 0.0001f, vPL);
					accLightColor += directLight->material.color * radianceP * world.AnyIntersection(shadowRay);
				}

				return accLightColor;
			}
			__device__ CudaColor<float> PointDirectSampling(
				ThreadData& thread,
				const CudaWorld& world,
				RayIntersection& intersection)
			{
				// Legend:
				// L - position of current light
				// P - point of intersetion
				// vN - surface normal

				CudaColor<float> accLightColor(0.0f, 0.0f, 0.0f, 1.0f);

				// [>] PointLights
				for (uint32_t index = 0u, tested = 0u;
					(index < world.pointLights.GetCapacity() && tested < world.pointLights.GetCount());
					++index)
				{
					const CudaPointLight* point_light = &world.pointLights[index];
					if (!point_light->Exist()) continue;
					++tested;


					// randomize point light position
					const cudaVec3<float> rndL = point_light->position + cudaVec3<float>(
						ckernel->GetRndNumbers().GetUnsignedUniform(thread) * 2.0f - 1.0f,
						ckernel->GetRndNumbers().GetUnsignedUniform(thread) * 2.0f - 1.0f,
						ckernel->GetRndNumbers().GetUnsignedUniform(thread) * 2.0f - 1.0f) * point_light->size;

					// vector from point to light position
					const cudaVec3<float> vPL = rndL - intersection.point;

					// distance factor (inverse square law)
					const float dPL = vPL.Length();
					const float d_factor = 1.0f / (dPL * dPL + 1.0f);

					// scattering factor
					const float sctr_factor = __expf(-dPL * intersection.ray.material->scattering);

					// calculate radiance at P
					const float radianceP = point_light->material.emittance * d_factor * sctr_factor;
					if (radianceP < 0.0001f) continue;	// unimportant light contribution

					// cast shadow ray and accumulate light contribution
					const CudaRay shadowRay(intersection.point, vPL, dPL);
					accLightColor += point_light->material.color * radianceP * world.AnyIntersection(shadowRay);
				}


				// [>] SpotLights
				for (uint32_t index = 0u, tested = 0u;
					(index < world.spotLights.GetCapacity() && tested < world.spotLights.GetCount());
					++index)
				{
					const CudaSpotLight* spotLight = &world.spotLights[index];
					if (!spotLight->Exist()) continue;
					++tested;

					// randomize spot light position
					const cudaVec3<float> rndL = spotLight->position + cudaVec3<float>(
						ckernel->GetRndNumbers().GetUnsignedUniform(thread) * 2.0f - 1.0f,
						ckernel->GetRndNumbers().GetUnsignedUniform(thread) * 2.0f - 1.0f,
						ckernel->GetRndNumbers().GetUnsignedUniform(thread) * 2.0f - 1.0f) * spotLight->size;

					// vector from point to light position
					const cudaVec3<float> vPL = rndL - intersection.point;

					// distance factor (inverse square law)
					const float dPL = vPL.Length();
					const float d_factor = 1.0f / (dPL * dPL + 1.0f);

					// scattering factor
					const float sctr_factor = __expf(-dPL * intersection.ray.material->scattering);

					// beam illumination
					float beamIllum = 1.0f;
					const float LP_dot_D = cudaVec3<float>::Similarity(-vPL, spotLight->direction);
					if (LP_dot_D < spotLight->cos_angle) beamIllum = 0.0f;
					else beamIllum = 1.0f;

					// calculate radiance at P
					const float radianceP = spotLight->material.emittance * d_factor * sctr_factor * beamIllum;
					if (radianceP < 0.0001f) continue;	// unimportant light contribution

					// cast shadow ray and calculate color contribution
					const CudaRay shadowRay(intersection.point, vPL, dPL);
					accLightColor += spotLight->material.color * radianceP * world.AnyIntersection(shadowRay);
				}


				// [>] DirectLights
				for (uint32_t index = 0u, tested = 0u;
					(index < world.directLights.GetCapacity() && tested < world.directLights.GetCount());
					++index)
				{
					const CudaDirectLight* directLight = &world.directLights[index];
					if (!directLight->Exist()) continue;
					++tested;

					// vector from point to direct light (reversed direction)
					const cudaVec3<float> vPL = SampleSphere(
						ckernel->GetRndNumbers().GetUnsignedUniform(thread),
						0.5f * (1.0f - __cosf(
							ckernel->GetRndNumbers().GetUnsignedUniform(thread) *
							directLight->angular_size)),
						-directLight->direction);

					// calculate light energy at P
					float energyAtP = directLight->material.emittance;
					if (energyAtP < 0.0001f) continue;	// unimportant light contribution

					// cast shadow ray and calculate color contribution
					CudaRay shadowRay(intersection.point, vPL);
					accLightColor += directLight->material.color * energyAtP * world.AnyIntersection(shadowRay);
				}

				return accLightColor;
			}

			__device__ void GenerateDiffuseRay(
				ThreadData& thread,
				RayIntersection& intersection)
			{
				cudaVec3<float> sample = CosineSampleHemisphere(
					ckernel->GetRndNumbers().GetUnsignedUniform(thread),
					ckernel->GetRndNumbers().GetUnsignedUniform(thread),
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
				ThreadData& thread,
				RayIntersection& intersection)
			{
				if (intersection.surface_material->glossiness > 0.0f)
				{
					const cudaVec3<float> vNd = SampleHemisphere(
						ckernel->GetRndNumbers().GetUnsignedUniform(thread),
						1.0f - __powf(
							ckernel->GetRndNumbers().GetUnsignedUniform(thread),
							intersection.surface_material->glossiness),
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

					GenerateSpecularRay(intersection);
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
			__device__ void GenerateTransmissiveRay(
				ThreadData& thread,
				RayIntersection& intersection)
			{
				if (intersection.behind_material->ior != intersection.ray.material->ior)
				{	// refraction ray

					const float cosi = fabsf(cudaVec3<float>::DotProduct(
						intersection.ray.direction, intersection.mapped_normal));

					// calculate sin^2 theta from Snell's law
					const float n1 = intersection.ray.material->ior;
					const float n2 = intersection.behind_material->ior;
					const float ratio = n1 / n2;
					const float sin2_t = ratio * ratio * (1.0f - cosi * cosi);

					if (sin2_t >= 1.0f)
					{	// TIR

						// calculate reflection vector
						cudaVec3<float> vR = ReflectVector(
							intersection.ray.direction,
							intersection.mapped_normal);

						// flip sample above surface if needed
						const float vR_dot_vN = cudaVec3<float>::DotProduct(vR, intersection.surface_normal);
						if (vR_dot_vN < 0.0f) vR += intersection.surface_normal * -2.0f * vR_dot_vN;

						// create new internal reflection CudaSceneRay
						new (&intersection.ray) CudaSceneRay(
							intersection.point + intersection.surface_normal * 0.0001f,
							vR,
							intersection.ray.material);
					}
					else
					{
						// calculate fresnel
						const float cost = sqrtf(1.0f - sin2_t);
						const float Rp = ((n1 * cosi) - (n2 * cost)) / ((n1 * cosi) + (n2 * cost));
						const float Rs = ((n2 * cosi) - (n1 * cost)) / ((n2 * cosi) + (n1 * cost));
						const float f = (Rs * Rs + Rp * Rp) / 2.0f;

						if (f < ckernel->GetRndNumbers().GetUnsignedUniform(thread))
						{	// transmission/refraction

							// calculate refraction direction
							const cudaVec3<float> vR = intersection.ray.direction * ratio +
								intersection.mapped_normal * (ratio * cosi - cost);

							// create new refraction CudaSceneRay
							new (&intersection.ray) CudaSceneRay(
								intersection.point - intersection.surface_normal * 0.0001f,
								vR,
								intersection.behind_material);
						}
						else
						{	// reflection

							// calculate reflection direction
							cudaVec3<float> vR = ReflectVector(
								intersection.ray.direction,
								intersection.mapped_normal);

							// flip sample above surface if needed
							const float vR_dot_vN = cudaVec3<float>::DotProduct(vR, intersection.surface_normal);
							if (vR_dot_vN < 0.0f) vR += intersection.surface_normal * -2.0f * vR_dot_vN;

							// create new reflection CudaSceneRay
							new (&intersection.ray) CudaSceneRay(
								intersection.point + intersection.surface_normal * 0.0001f,
								vR,
								intersection.ray.material);
						}
					}
				}
				else
				{	// transparent ray

					cudaVec3<float> vD;

					if (intersection.behind_material->glossiness > 0.0f)
					{
						vD = SampleSphere(
							ckernel->GetRndNumbers().GetUnsignedUniform(thread),
							1.0f - __powf(
								ckernel->GetRndNumbers().GetUnsignedUniform(thread),
								intersection.behind_material->glossiness),
							intersection.ray.direction);

						const float vS_dot_vN = cudaVec3<float>::DotProduct(vD, -intersection.surface_normal);
						if (vS_dot_vN < 0.0f) vD += -intersection.surface_normal * -2.0f * vS_dot_vN;
					}
					else
					{
						vD = intersection.ray.direction;
					}

					new (&intersection.ray) CudaSceneRay(
						intersection.point - intersection.surface_normal * 0.0001f,
						vD,
						intersection.behind_material);
				}
			}



			// [>] Tone mapping
			__global__ void ToneMap(
				CudaGlobalKernel* const global_kernel,
				CudaWorld* const world,
				const int camera_id)
			{
				CudaCamera* const camera = &world->cameras[camera_id];

				// calculate thread position
				const uint32_t thread_x = blockIdx.x * blockDim.x + threadIdx.x;
				const uint32_t thread_y = blockIdx.y * blockDim.y + threadIdx.y;
				if (thread_x >= camera->GetWidth() || thread_y >= camera->GetHeight()) return;

				// average sample color by dividing by number of samples
				CudaColor<float> samplingColor =
					camera->GetSample(thread_x, thread_y);
				samplingColor /= camera->GetPassesCount();

				// tone map sample color
				camera->SetFinalPixel(global_kernel->GetRenderIdx(),
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
				const uint32_t thread_x = blockIdx.x * blockDim.x + threadIdx.x;
				const uint32_t thread_y = blockIdx.y * blockDim.y + threadIdx.y;
				if (thread_x >= camera->GetWidth() || thread_y >= camera->GetHeight()) return;

				// reset sample buffer
				camera->SetSample(CudaColor<float>(0.0f, 0.0f, 0.0f, FLT_EPSILON), thread_x, thread_y);

				// TODO: reset tracing paths
			}
			__global__ void CudaCameraUpdateSamplesNumber(
				CudaWorld* const world,
				const int camera_id,
				bool reset_flag)
			{
				CudaCamera* const camera = &world->cameras[camera_id];
				if (reset_flag)	camera->GetPassesCount() = 1u;
				else			camera->GetPassesCount() += 1u;
			}
		}
	}
}