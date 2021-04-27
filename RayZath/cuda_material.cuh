#ifndef CUDA_MATERIAL_H
#define CUDA_MATERIAL_H

#include "cuda_render_parts.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		struct CudaMaterial : public WithExistFlag
		{
		private:
			CudaColor<float> color;

			float reflectance;
			float glossiness;

			float transmittance;
			float ior;

			float emittance;
			float scattering;

			const CudaTexture* texture;

		public:
			__host__ __device__ CudaMaterial(
				const CudaColor<float>& color = CudaColor<float>(1.0f, 1.0f, 1.0f, 1.0f),
				const float& reflectance = 0.0f,
				const float& glossiness = 0.0f,
				const float& transmitance = 1.0f,
				const float& ior = 1.0f,
				const float& emittance = 0.0f,
				const float& scattering = 0.0f)
				: color(color)
				, reflectance(reflectance)
				, glossiness(glossiness)
				, transmittance(transmitance)
				, ior(ior)
				, emittance(emittance)
				, scattering(scattering)
				, texture(nullptr)
			{}

			__host__ CudaMaterial& operator=(const Material& hMaterial);

			__host__ void Reconstruct(
				const CudaWorld& hCudaWorld,
				const Handle<Material>& hMaterial,
				cudaStream_t& mirror_stream);


		public:
			__host__ void SetColor(const CudaColor<float>& color)
			{
				this->color = color;
			}
			__host__ void SetEmittance(const float& emittance)
			{
				this->emittance = emittance;
			}
			__host__ void SetTexture(const CudaTexture* texture)
			{
				this->texture = texture;
			}

			__device__ CudaColor<float> GetColor() const
			{
				return color;
			}
			__device__ CudaColor<float> GetColor(const CudaTexcrd& texcrd) const
			{
				if (texture) return texture->Fetch(texcrd);
				else return GetColor();
			}
			__device__ const float& GetReflectance() const
			{
				return reflectance;
			}
			__device__ const float& GetGlossiness() const
			{
				return glossiness;
			}
			__device__ const float& GetTransmittance() const
			{
				return transmittance;
			}
			__device__ const float& GetIOR() const
			{
				return ior;
			}
			__device__ const float& GetEmittance() const
			{
				return emittance;
			}
			__device__ const float& GetScattering() const
			{
				return scattering;
			}


			__device__ bool SampleDirect(
				ThreadData& thread,
				const CudaConstantKernel* ckernel) const
			{
				if (transmittance > 0.0f) return false;
				if (ckernel->GetRndNumbers().GetUnsignedUniform(thread) >
					reflectance)
				{
					return true;
				}
				else
				{
					return false;
				}
			}

			__device__ void GenerateNextRay(
				ThreadData& thread,
				RayIntersection& intersection,
				const CudaConstantKernel* ckernel) const
			{
				if (intersection.surface_material->transmittance > 0.0f)
				{	// ray fallen into material/object
					GenerateTransmissiveRay(thread, intersection, ckernel);
				}
				else
				{	// ray is reflected from sufrace

					if (ckernel->GetRndNumbers().GetUnsignedUniform(thread) >
						intersection.surface_material->reflectance)
					{	// diffuse reflection
						GenerateDiffuseRay(thread, intersection, ckernel);
					}
					else
					{	// glossy reflection
						GenerateGlossyRay(thread, intersection, ckernel);
					}
				}
			}


		private:
			__device__ void GenerateDiffuseRay(
				ThreadData& thread,
				RayIntersection& intersection,
				const CudaConstantKernel* ckernel) const
			{
				vec3f sample = CosineSampleHemisphere(
					ckernel->GetRndNumbers().GetUnsignedUniform(thread),
					ckernel->GetRndNumbers().GetUnsignedUniform(thread),
					intersection.mapped_normal);
				sample.Normalize();

				// flip sample above surface if needed
				const float vR_dot_vN = vec3f::Similarity(sample, intersection.surface_normal);
				if (vR_dot_vN < 0.0f) sample += intersection.surface_normal * -2.0f * vR_dot_vN;

				new (&intersection.ray) CudaSceneRay(
					intersection.point + intersection.surface_normal * 0.0001f,
					sample,
					intersection.ray.material);
			}
			__device__ void GenerateSpecularRay(
				RayIntersection& intersection) const
			{
				vec3f reflect = ReflectVector(
					intersection.ray.direction,
					intersection.mapped_normal);

				// flip sample above surface if needed
				const float vR_dot_vN = vec3f::Similarity(reflect, intersection.surface_normal);
				if (vR_dot_vN < 0.0f) reflect += intersection.surface_normal * -2.0f * vR_dot_vN;

				new (&intersection.ray) CudaSceneRay(
					intersection.point + intersection.surface_normal * 0.0001f,
					reflect, intersection.ray.material);
			}
			__device__ void GenerateGlossyRay(
				ThreadData& thread,
				RayIntersection& intersection,
				const CudaConstantKernel* ckernel) const
			{
				if (intersection.surface_material->glossiness > 0.0f)
				{
					const vec3f vNd = SampleHemisphere(
						ckernel->GetRndNumbers().GetUnsignedUniform(thread),
						1.0f - cui_powf(
							ckernel->GetRndNumbers().GetUnsignedUniform(thread),
							intersection.surface_material->glossiness),
						intersection.mapped_normal);

					// calculate reflection direction
					vec3f vR = ReflectVector(
						intersection.ray.direction,
						vNd);

					// reflect sample above surface if needed
					const float vR_dot_vN = vec3f::Similarity(vR, intersection.surface_normal);
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

				/*GlossySpecular::sample_f(const ShadeRec& sr,
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
				}*/
			}
			__device__ void GenerateTransmissiveRay(
				ThreadData& thread,
				RayIntersection& intersection,
				const CudaConstantKernel* ckernel) const
			{
				if (intersection.behind_material->ior != intersection.ray.material->ior)
				{	// refraction ray

					const float cosi = fabsf(vec3f::DotProduct(
						intersection.ray.direction, intersection.mapped_normal));

					// calculate sin^2 theta from Snell's law
					const float& n1 = intersection.ray.material->ior;
					const float& n2 = intersection.behind_material->ior;
					const float ratio = n1 / n2;
					const float sin2_t = ratio * ratio * (1.0f - cosi * cosi);

					if (sin2_t >= 1.0f)
					{	// TIR

						// calculate reflection vector
						vec3f vR = ReflectVector(
							intersection.ray.direction,
							intersection.mapped_normal);

						// flip sample above surface if needed
						const float vR_dot_vN = vec3f::DotProduct(vR, intersection.surface_normal);
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
							const vec3f vR = intersection.ray.direction * ratio +
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
							vec3f vR = ReflectVector(
								intersection.ray.direction,
								intersection.mapped_normal);

							// flip sample above surface if needed
							const float vR_dot_vN = vec3f::DotProduct(vR, intersection.surface_normal);
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

					vec3f vD;

					if (intersection.behind_material->glossiness > 0.0f)
					{
						vD = SampleSphere(
							ckernel->GetRndNumbers().GetUnsignedUniform(thread),
							1.0f - cui_powf(
								ckernel->GetRndNumbers().GetUnsignedUniform(thread),
								intersection.behind_material->glossiness),
							intersection.ray.direction);

						const float vS_dot_vN = vec3f::DotProduct(vD, -intersection.surface_normal);
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
		};
	}
}

#endif // !CUDA_MATERIAL_H