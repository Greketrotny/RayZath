#include "cpu_engine_kernel.hpp"
#include "engine_parts.hpp"

#include <numbers>

#include <iostream>

namespace RayZath::Engine::CPU
{
	void Kernel::setWorld(World& world)
	{
		mp_world = &world;
	}

	Graphics::ColorF Kernel::renderFirstPass(
		Camera& camera,
		CameraContext& context,
		const Math::vec2ui32 pixel,
		RNG& rng,
		const RenderConfig& config) const
	{
		RZAssertCore(bool(mp_world), "mp_world was nullptr");

		SceneRay ray{};
		ray.material = &mp_world->material();
		generateSimpleRay(camera, ray, pixel);

		TracingState tracing_state{Graphics::ColorF(0.0f), 0u};
		auto result{traceRay(tracing_state, ray, rng, config)};
		const bool path_continues = tracing_state.path_depth < config.tracing().maxDepth();

		// set depth
		camera.depthBuffer().Value(pixel.x, pixel.y) = ray.near_far.y;
		// set intersection point (for spatio-temporal reprojection)
		//camera.spaceBuffer().Value(pixel.x, pixel.y) = ray.origin + ray.direction * ray.near_far.y;

		// set color value
		tracing_state.final_color.alpha = float(!path_continues);
		context.m_image.Value(pixel.x, pixel.y) = tracing_state.final_color;

		if (path_continues)
		{
			result.repositionRay(ray);
		}
		else
		{
			// generate camera ray
			generateAntialiasedRay(camera, ray, pixel, rng);
			ray.material = &mp_world->material();
			ray.color = Graphics::ColorF(1.0f);
		}

		context.setRay(pixel, ray);
		context.m_path_depth.Value(pixel.x, pixel.y) = path_continues ? tracing_state.path_depth : 0u;

		return tracing_state.final_color;
	}
	Graphics::ColorF Kernel::renderCumulativePass(
		const Camera& camera,
		CameraContext& context,
		const Math::vec2ui32 pixel,
		RNG& rng,
		const RenderConfig& config) const
	{
		RZAssert(bool(mp_world), "mp_world was nullptr");
				
		TracingState tracing_state(
			Graphics::ColorF(0.0f),
			context.m_path_depth.Value(pixel.x, pixel.y));

		// trace ray through scene
		SceneRay ray{context.getRay(pixel)};
		if (tracing_state.path_depth == 0) ray.near_far = camera.nearFar();
		const auto result{traceRay(tracing_state, ray, rng, config)};
		const bool path_continues = tracing_state.path_depth < config.tracing().maxDepth();

		// append additional light contribution passing along traced ray
		auto value{context.m_image.Value(pixel.x, pixel.y)};
		value.red += tracing_state.final_color.red;
		value.green += tracing_state.final_color.green;
		value.blue += tracing_state.final_color.blue;
		value.alpha += float(!path_continues);
		context.m_image.Value(pixel.x, pixel.y) = value;

		if (path_continues)
		{
			result.repositionRay(ray);
		}
		else
		{
			// generate camera ray
			generateAntialiasedRay(camera, ray, pixel, rng);
			ray.material = &mp_world->material();
			ray.color = Graphics::ColorF(1.0f);
		}

		context.setRay(pixel, ray);
		context.m_path_depth.Value(pixel.x, pixel.y) = path_continues ? tracing_state.path_depth : 0u;

		return value;
	}
	void Kernel::rayCast(Camera& camera) const
	{
		RangedRay ray;
		generateSimpleRay(camera, ray, camera.getRayCastPixel());
		const auto depth = camera.depthBuffer().Value(camera.getRayCastPixel().x, camera.getRayCastPixel().y);
		ray.near_far.x = depth * 0.99f;
		ray.near_far.y = depth * 1.01f;

		std::tie(camera.m_raycasted_instance, camera.m_raycasted_material) = worldRayCast(ray);
	}

	TracingResult Kernel::traceRay(TracingState& tracing_state, SceneRay& ray, RNG& rng, const RenderConfig& config) const
	{
		SurfaceProperties surface(&mp_world->material());
		auto any_hit = closestIntersection(ray, surface);

		surface.color = fetchColor(*surface.surface_material, surface.texcrd);
		surface.emission = fetchEmission(*surface.surface_material, surface.texcrd);

		if (surface.emission > 0.0f)
		{	// intersection with emitting object

			tracing_state.final_color +=
				ray.color *
				surface.color *
				surface.emission;
		}

		if (!any_hit)
		{	// nothing has been hit - terminate path

			tracing_state.endPath();
			return TracingResult{};
		}
		++tracing_state.path_depth;


		// Fetch metalness  and roughness from surface material (for BRDF and next even estimation)
		surface.metalness = fetchMetalness(*surface.surface_material, surface.texcrd);
		surface.roughness = fetchRoughness(*surface.surface_material, surface.texcrd);

		// calculate fresnel and reflectance ratio (for BRDF and next ray generation)
		surface.fresnel = fresnelSpecularRatio(
			surface.mapped_normal,
			ray.direction,
			ray.material->ior(),
			surface.behind_material->ior(),
			surface.refraction_factors);
		surface.reflectance = lerp(surface.fresnel, 1.0f, surface.metalness);

		TracingResult result;
		// sample direction (importance sampling) (for next ray generation and direct light sampling (MIS))
		result.next_direction = sampleDirection(ray, surface, rng);
		// find intersection point (for direct sampling and next ray generation)
		result.point =
			ray.origin + ray.direction * ray.near_far.y +	// origin + ray vector
			surface.normal * (0.0001f * ray.near_far.y);	// normal nodge

		// Direct sampling
		if (true) // sample direct light?
		{
			// sample direct light
			const Graphics::ColorF direct_illumination = directIllumination(
				ray, result, surface,
				rng, 
				config);

			// add direct light
			tracing_state.final_color +=
				direct_illumination * // incoming radiance from lights
				ray.color * // ray color mask
				lerp(Graphics::ColorF(1.0f), surface.color, surface.metalness); // metalic factor
		}

		ray.color.Blend(ray.color * surface.color, surface.tint_factor);
		return result;
	}

	void Kernel::generateSimpleRay(const Camera& camera, RangedRay& ray, const Math::vec2ui32& pixel) const
	{
		using namespace Math;
		ray.origin = vec3f32(0.0f);

		// ray to screen deflection
		const float tana = std::tanf(camera.fov().value() * 0.5f);
		const vec2f32 dir =
			(((vec2f32(pixel) + vec2f32(0.5f)) /
				vec2f32(camera.resolution())) -
				vec2f32(0.5f)) *
			vec2f(tana, -tana / camera.aspectRatio());
		ray.direction.x = dir.x;
		ray.direction.y = dir.y;
		ray.direction.z = 1.0f;

		// camera transformation
		ray.origin = camera.coordSystem().transformForward(ray.origin);
		ray.origin += camera.position();
		ray.direction = camera.coordSystem().transformForward(ray.direction);
		ray.direction.Normalize();

		// apply near/far clipping plane
		ray.near_far = camera.nearFar();
	}
	void Kernel::generateAntialiasedRay(
		const Camera& camera,
		RangedRay& ray,
		const Math::vec2ui32& pixel,
		RNG& rng) const
	{
		using namespace Math;

		// ray to screen deflection
		const float tana = std::tanf(camera.fov().value() * 0.5f);
		const vec2f dir =
			(((vec2f(pixel) + vec2f(0.5f)) /
				vec2f(camera.resolution())) -
				vec2f(0.5f)) *
			vec2f(tana, -tana / camera.aspectRatio());
		ray.direction.x = dir.x;
		ray.direction.y = dir.y;
		ray.direction.z = 1.0f;

		// pixel position distortion (antialiasing)
		ray.direction.x +=
			((0.5f / float(camera.resolution().x)) * (rng.signedUniform()));
		ray.direction.y +=  // this --v-- should be x
			((0.5f / float(camera.resolution().x)) * (rng.signedUniform()));

		// focal point
		const vec3f focalPoint = ray.direction * camera.focalDistance();

		// aperture distortion
		const float aperture_angle = rng.unsignedUniform() * 2.0f * std::numbers::pi_v<float>;
		const float aperture_sample = std::sqrtf(rng.unsignedUniform()) * camera.aperture();
		ray.origin = vec3f(
			aperture_sample * std::sinf(aperture_angle),
			aperture_sample * std::cosf(aperture_angle),
			0.0f);

		// depth of field ray
		ray.direction = focalPoint - ray.origin;

		// camera transformation
		ray.origin = camera.coordSystem().transformForward(ray.origin);
		ray.origin += camera.position();
		ray.direction = camera.coordSystem().transformForward(ray.direction);
		ray.direction.Normalize();

		// apply near/far clipping plane
		ray.near_far = camera.nearFar();
	}
	
	void Kernel::traverseWorld(const tree_node_t& node, RangedRay& ray, TraversalResult& traversal) const
	{
		if (node.isLeaf())
		{
			for (const auto& object : node.objects())
			{
				if (!object) continue;
				closestIntersection(object, ray, traversal);
			}
		}
		else
		{
			const auto& first_child = node.children()->first;
			if (first_child.boundingBox().rayIntersection(ray))
			{
				traverseWorld(first_child, ray, traversal);
			}
			const auto& second_child = node.children()->second;
			if (second_child.boundingBox().rayIntersection(ray))
			{
				traverseWorld(second_child, ray, traversal);
			}
		}
	}

	bool Kernel::closestIntersection(RangedRay& ray, SurfaceProperties& surface) const
	{
		const auto& instances = mp_world->container<ObjectType::Instance>();
		if (instances.empty()) return false;
		if (!instances.root().boundingBox().rayIntersection(ray)) return false;
		
		TraversalResult traversal;
		traverseWorld(instances.root(), ray, traversal);

		const bool found = traversal.closest_instance != nullptr;
		if (found) analyzeIntersection(*traversal.closest_instance, traversal, surface);
		else
		{
			// texcrd of the sky sphere
			surface.texcrd = Texcrd(
				-(0.5f + (atan2f(ray.direction.z, ray.direction.x) / (std::numbers::pi_v<float> *2.0f))),
				0.5f + (asinf(ray.direction.y) / std::numbers::pi_v<float>));
		}
		return found;
	}
	void Kernel::closestIntersection(
		const Handle<Instance>& instance_handle, 
		RangedRay& ray, 
		TraversalResult& traversal) const
	{
		const auto& instance = *instance_handle.accessor()->get();
		if (!instance.boundingBox().rayIntersection(ray)) return;

		RangedRay local_ray = ray;
		instance.transformation().transformG2L(local_ray);

		const float length_factor = local_ray.direction.Magnitude();
		local_ray.near_far *= length_factor;
		local_ray.direction.Normalize();

		const auto& mesh = instance.mesh();
		if (!mesh) return;
		const auto* const closest_triangle = traversal.closest_triangle;
		traversal.closest_triangle = nullptr;
		
		closestIntersection(*mesh, local_ray, traversal);
		if (traversal.closest_triangle)
		{
			traversal.closest_instance = &instance;
			traversal.instance_idx = instance_handle.accessor()->idx();
			ray.near_far = local_ray.near_far / length_factor;
		}
		else
		{
			traversal.closest_triangle = closest_triangle;
		}
	}
	void Kernel::closestIntersection(const Mesh& mesh, RangedRay& ray, TraversalResult& traversal) const
	{
		auto traverse = [&](const auto& self, const auto& node) {
			if (!node.boundingBox().rayIntersection(ray)) return;

			if (node.isLeaf())
			{
				for (const auto& triangle : node.objects())
				{
					triangle->closestIntersection(ray, traversal, mesh);
				}
			}
			else
			{
				RZAssertCore(node.children(), "If not leaf, should have children.");
				self(self, node.children()->first);
				self(self, node.children()->second);
			}
		};

		traverse(traverse, mesh.triangles().getBVH().rootNode());
	}
	
	void Kernel::analyzeIntersection(
		const Instance& instance, 
		TraversalResult& traversal, 
		SurfaceProperties& surface) const
	{
		const auto& material = instance.material(traversal.closest_triangle->material_id);
		if (!material) surface.surface_material = &mp_world->defaultMaterial();
		else surface.surface_material = material.accessor()->get();
		if (traversal.external) surface.behind_material = surface.surface_material;
		else surface.behind_material = &mp_world->material();

		// calculate texture coordinates
		RZAssertCore(instance.mesh(), "instance had no mesh");
		if (traversal.closest_triangle->texcrds != Mesh::ids_unused)
			surface.texcrd = traversal.closest_triangle->texcrdFromBarycenter(traversal.barycenter, *instance.mesh());

		const float external_factor = static_cast<float>(traversal.external) * 2.0f - 1.0f;

		// calculate mapped normal
		surface.mapped_normal = traversal.closest_triangle->normals != Mesh::ids_unused ?
			traversal.closest_triangle->averageNormal(traversal.barycenter, *instance.mesh()) :
			traversal.closest_triangle->normal;
		if (surface.surface_material->map<ObjectType::NormalMap>())
		{			
			traversal.closest_triangle->mapNormal(
				Graphics::ColorF(surface.surface_material->map<ObjectType::NormalMap>()->fetch(surface.texcrd)),
				surface.mapped_normal,
				*instance.mesh(),
				instance.transformation().scale());
			instance.transformation().transformL2GNoScale(surface.mapped_normal);
		}
		else
		{
			instance.transformation().transformL2G(surface.mapped_normal);
		}
		surface.mapped_normal.Normalize();
		surface.mapped_normal *= external_factor;

		surface.normal = traversal.closest_triangle->normal * external_factor;
		instance.transformation().transformL2G(surface.normal);
		surface.normal.Normalize();
	}


	Graphics::ColorF Kernel::anyIntersection(const RangedRay& ray) const
	{
		const auto& instances = mp_world->container<ObjectType::Instance>();
		if (instances.empty()) return Graphics::ColorF(0.0f);
		if (!instances.root().boundingBox().rayIntersection(ray)) return Graphics::ColorF(1.0f);

		Graphics::ColorF shadow_mask(1.0f);
		auto traverseWorld = [&](const auto& self, const tree_node_t& node)
		{
			if (node.isLeaf())
			{
				for (const auto& object : node.objects())
				{
					if (!object) continue;
					shadow_mask *= anyIntersection(*object, ray);
					if (shadow_mask.alpha < 1.0e-4f) return;
				}
			}
			else
			{
				const auto& first_child = node.children()->first;
				if (first_child.boundingBox().rayIntersection(ray))
				{
					self(self, first_child);
					if (shadow_mask.alpha < 1.0e-4f) return;
				}
				const auto& second_child = node.children()->second;
				if (second_child.boundingBox().rayIntersection(ray))
				{
					self(self, second_child);
				}
			}
		};
		
		traverseWorld(traverseWorld, instances.root());
		return shadow_mask;
	}
	Graphics::ColorF Kernel::anyIntersection(const Instance& instance, const RangedRay& ray) const
	{
		// [>] check ray intersection with bounding_box
		if (!instance.boundingBox().rayIntersection(ray))
			return Graphics::ColorF(1.0f);

		// [>] transpose objectSpaceRay
		RangedRay local_ray = ray;
		instance.transformation().transformG2L(local_ray);
		local_ray.near_far *= local_ray.direction.Magnitude();
		local_ray.direction.Normalize();

		if (!instance.mesh()) return Graphics::ColorF(1.0f);
		return anyIntersection(*instance.mesh(), local_ray);
	}
	Graphics::ColorF Kernel::anyIntersection(const Mesh& mesh, const RangedRay& ray) const
	{
		Math::vec2f32 barycenter;
		Graphics::ColorF shadow_mask(1.0f);		

		auto traverse = [&](const auto& self, const auto& node) {
			if (shadow_mask.alpha < 1.0e-4f) return;
			if (!node.boundingBox().rayIntersection(ray)) return;

			if (node.isLeaf())
			{
				for (const auto& triangle : node.objects())
				{
					if (triangle->anyIntersection(ray, barycenter, mesh))
					{
						// TODO: texture fetch
						shadow_mask *= 0.0f;
						return;
					}
				}
			}
			else
			{
				RZAssertCore(node.children(), "If not leaf, should have children.");
				self(self, node.children()->first);
				self(self, node.children()->second);
			}
		};

		traverse(traverse, mesh.triangles().getBVH().rootNode());
		return shadow_mask;
	}

	std::tuple<Handle<Instance>, Handle<Material>> Kernel::worldRayCast(RangedRay& ray) const
	{
		const auto& instances = mp_world->container<ObjectType::Instance>();
		if (instances.empty()) return {};
		if (!instances.root().boundingBox().rayIntersection(ray)) return {};

		TraversalResult traversal;
		traverseWorld(instances.root(), ray, traversal);

		Handle<Instance> instance{};
		Handle<Material> material{};
		if (traversal.closest_instance)
		{
			instance = instances[traversal.instance_idx];
			material = traversal.closest_instance->material(traversal.closest_triangle->material_id);
		}

		return {std::move(instance), std::move(material)};
	}

	// ~~~~~~~~ material functions ~~~~~~~~
	// texture/map fetching
	Graphics::ColorF Kernel::fetchColor(const Material& material, const Texcrd& texcrd) const
	{
		Graphics::ColorF color = Graphics::ColorF(material.color());
		if (const auto& texture = material.map<ObjectType::Texture>(); texture)
			color = Graphics::ColorF(texture->fetch(texcrd));
		color.alpha = 1.0f - color.alpha;
		return color;
	}
	float Kernel::fetchMetalness(const Material& material, const Texcrd& texcrd) const
	{
		if (const auto& metalness_map = material.map<ObjectType::MetalnessMap>(); metalness_map)
		{
			using value_t = std::decay_t<decltype(metalness_map->fetch(texcrd))>;
			return float(metalness_map->fetch(texcrd)) / float(std::numeric_limits<value_t>::max());
		}
			
		return material.metalness();
	}
	float Kernel::fetchEmission(const Material& material, const Texcrd& texcrd) const
	{
		if (const auto& emission_map = material.map<ObjectType::EmissionMap>(); emission_map)
			return emission_map->fetch(texcrd);
		return material.emission();
	}
	float Kernel::fetchRoughness(const Material& material, const Texcrd& texcrd) const
	{
		if (const auto& roughness_map = material.map<ObjectType::RoughnessMap>(); roughness_map)
		{
			using value_t = std::decay_t<decltype(roughness_map->fetch(texcrd))>;
			return float(roughness_map->fetch(texcrd)) / float(std::numeric_limits<value_t>::max());
		}
		return material.roughness();
	}

	bool Kernel::applyScattering(const Material& material, SceneRay& ray, SurfaceProperties& surface, RNG& rng) const
	{
		if (const auto scattering = material.scattering(); scattering > 1.0e-4f)
		{
			const float scatter_distance = (-std::logf(rng.unsignedUniform() + 1.0e-4f)) / scattering;
			if (scatter_distance < ray.near_far.y)
			{
				ray.near_far.y = scatter_distance;
				surface.surface_material = &material;
				surface.behind_material = &material;
				surface.normal = surface.mapped_normal = ray.direction;
				return true;
			}
		}
		return false;
	}

	float Kernel::BRDF(
		const RangedRay& ray, 
		const SurfaceProperties& surface, 
		const Math::vec3f32& vPL) const
	{
		if (surface.surface_material->scattering() > 0.0f) return 1.0f;

		const float vN_dot_vO = Math::vec3f32::DotProduct(surface.mapped_normal, vPL);
		if (vN_dot_vO <= 0.0f) return 0.0f;
		const float vN_dot_vI = Math::vec3f32::DotProduct(surface.mapped_normal, -ray.direction);
		if (vN_dot_vI <= 0.0f) return 0.0f;

		const Math::vec3f32 vI_half_vO = halfwayVector(ray.direction, vPL);

		const float nornal_distribution = NDF(surface.mapped_normal, vI_half_vO, surface.roughness);
		const float atten_i = attenuation(vN_dot_vI, surface.roughness);
		const float atten_o = attenuation(vN_dot_vO, surface.roughness);
		const float attenuation = atten_i * atten_o;

		const float diffuse = vN_dot_vO * float(surface.color.alpha == 0.0f);
		const float specular = nornal_distribution * attenuation / (vN_dot_vI * vN_dot_vO);

		return lerp(diffuse, specular * vN_dot_vO, surface.reflectance);
	}
	Graphics::ColorF Kernel::BRDFColor(const SurfaceProperties& surface) const
	{
		return lerp(surface.color, Graphics::ColorF(1.0f), surface.reflectance);
	}

	float Kernel::NDF(const Math::vec3f32 vN, const Math::vec3f32 vH, const float roughness) const
	{
		const float vN_dot_vH = Math::vec3f32::DotProduct(vN, vH);
		const float b = (vN_dot_vH * vN_dot_vH) * (roughness - 1.0f) + 1.0001f;
		return (roughness + 1.0e-5f) / (b * b);
	}
	float Kernel::attenuation(const float cos_angle, const float roughness) const
	{
		return cos_angle / ((cos_angle * (1.0f - roughness)) + roughness);
	}

	Math::vec3f32 Kernel::sampleDirection(SceneRay& ray, SurfaceProperties& surface, RNG& rng) const
	{
		if (surface.color.alpha > 0.0f)
		{	// transmission
			if (surface.surface_material->scattering() > 0.0f)
			{
				return sampleScatteringDirection(ray, surface, rng);
			}
			else
			{
				return sampleTransmissionDirection(ray, surface, rng);
			}
		}
		else
		{	// reflection

			if (rng.unsignedUniform() > surface.reflectance)
			{	// diffuse reflection
				return sampleDiffuseDirection(surface, rng);
			}
			else
			{	// glossy reflection
				return sampleGlossyDirection(ray, surface, rng);
			}
		}
	}
	Math::vec3f32 Kernel::sampleDiffuseDirection(SurfaceProperties& surface, RNG& rng) const
	{
		Math::vec3f vO = cosineSampleHemisphere(
			rng.unsignedUniform(),
			rng.unsignedUniform(),
			surface.mapped_normal);

		// flip sample above surface if needed
		const float vR_dot_vN = Math::vec3f32::Similarity(vO, surface.normal);
		if (vR_dot_vN < 0.0f) vO += surface.normal * -2.0f * vR_dot_vN;

		surface.tint_factor = 1.0f;
		return vO;
	}
	Math::vec3f32 Kernel::sampleGlossyDirection(SceneRay& ray, SurfaceProperties& surface, RNG& rng) const
	{
		const Math::vec3f32 vH = sampleHemisphere(
			rng.unsignedUniform(),
			1.0f - std::powf(
				rng.unsignedUniform() + 1.0e-5f,
				surface.roughness),
			surface.mapped_normal);

		// calculate reflection direction
		Math::vec3f32 vO = reflectVector(ray.direction, vH);

		// reflect sample above surface if needed
		const float vR_dot_vN = Math::vec3f32::Similarity(vO, surface.normal);
		if (vR_dot_vN < 0.0f) vO += surface.normal * -2.0f * vR_dot_vN;

		surface.tint_factor = surface.metalness;
		return vO;
	}
	Math::vec3f32 Kernel::sampleTransmissionDirection(SceneRay& ray, SurfaceProperties& surface, RNG& rng) const
	{
		if (surface.fresnel < rng.unsignedUniform())
		{	// transmission

			const Math::vec3f32 vO = ray.direction * surface.refraction_factors.x +
				surface.mapped_normal * surface.refraction_factors.y;

			ray.material = surface.behind_material;
			surface.normal.Reverse();
			surface.tint_factor = 1.0f;
			return vO;
		}
		else
		{	// reflection

			Math::vec3f32 vO = reflectVector(ray.direction, surface.mapped_normal);

			// flip sample above surface if needed
			const float vR_dot_vN = Math::vec3f32::DotProduct(vO, surface.normal);
			if (vR_dot_vN < 0.0f) vO += surface.normal * -2.0f * vR_dot_vN;

			surface.tint_factor = surface.metalness;
			return vO;
		}
	}
	Math::vec3f32 Kernel::sampleScatteringDirection(SceneRay& ray, SurfaceProperties& surface, RNG& rng) const
	{
		// generate scatter direction
		const Math::vec3f32 vO = sampleSphere(rng.unsignedUniform(), rng.unsignedUniform(), ray.direction);
		surface.tint_factor = surface.metalness;
		return vO;
	}

	// ~~~~~~~~ direct sampling ~~~~~~~~
	Graphics::ColorF Kernel::spotLightSampling(
		const SceneRay& ray,
		const TracingResult& result,
		const SurfaceProperties& surface,
		const float vS_pdf,
		RNG& rng,
		const RenderConfig& config) const
	{
		const auto& lights = mp_world->container<ObjectType::SpotLight>();
		const uint32_t light_count = lights.count();
		const uint32_t sample_count = config.lightSampling().spotLight();

		Graphics::ColorF total_light(0.0f);
		if (light_count == 0) return total_light;
		for (uint32_t i = 0u; i < sample_count; ++i)
		{
			const auto& light = *lights[uint32_t(rng.unsignedUniform() * light_count)];

			// sample light
			float Se = 0.0f;
			const Math::vec3f32 vPL = spotLightSampleDirection(light, result.point, result.next_direction, Se, rng);
			const float dPL = vPL.Magnitude();

			const float brdf = BRDF(ray, surface, vPL / dPL);
			if (brdf < 1.0e-4f) continue;
			const Graphics::ColorF brdf_color = BRDFColor(surface);
			const float solid_angle = spotLightSolidAngle(light, dPL);
			const float sctr_factor = std::expf(-dPL * ray.material->scattering());

			// beam illumination
			const float beamIllum = spotLightBeamIllumination(light, vPL);
			if (beamIllum < 1.0e-4f)
				continue;

			// calculate radiance at P
			const float L_pdf = 1.0f / solid_angle;
			const float vSw = vS_pdf / (vS_pdf + L_pdf);
			const float Lw = 1.0f - vSw;
			const float Le = light.emission() * solid_angle * brdf;
			const float radiance = (Le * Lw + Se * vSw) * sctr_factor * beamIllum;
			if (radiance < 1.0e-4f) continue;	// unimportant light contribution

			// cast shadow ray and calculate color contribution
			const RangedRay shadowRay(result.point, vPL, Math::vec2f32(0.0f, dPL));
			const Graphics::ColorF V_PL = anyIntersection(shadowRay);
			total_light +=
				Graphics::ColorF(light.color()) *
				brdf_color *
				radiance *
				V_PL * V_PL.alpha;
		}

		const float pdf = sample_count / float(light_count);
		return total_light / pdf;
	}
	Graphics::ColorF Kernel::directLightSampling(
		const SceneRay& ray,
		const TracingResult& result,
		const SurfaceProperties& surface,
		const float vS_pdf,
		RNG& rng,
		const RenderConfig& config) const
	{
		const auto& lights = mp_world->container<ObjectType::DirectLight>();
		const uint32_t light_count = lights.count();
		const uint32_t sample_count = config.lightSampling().directLight();

		Graphics::ColorF total_light(0.0f);
		if (light_count == 0) return total_light;
		for (uint32_t i = 0u; i < sample_count; ++i)
		{
			const auto& light = *lights[uint32_t(rng.unsignedUniform() * light_count)];

			// sample light
			float Se = 0.0f;
			const Math::vec3f32 vPL = directLightSampleDirection(light, result.next_direction, Se, rng);

			const float brdf = BRDF(ray, surface, vPL.Normalized());
			const Graphics::ColorF brdf_color = BRDFColor(surface);
			const float solid_angle = directLightSolidAngle(light);

			// calculate radiance at P
			const float L_pdf = 1.0f / solid_angle;
			const float vSw = vS_pdf / (vS_pdf + L_pdf);
			const float Lw = 1.0f - vSw;
			const float Le = light.emission() * solid_angle * brdf;
			const float radiance = (Le * Lw + Se * vSw);
			if (radiance < 1.0e-4f) continue;	// unimportant light contribution

			// cast shadow ray and calculate color contribution
			const RangedRay shadowRay(result.point, vPL);
			const Graphics::ColorF V_PL = anyIntersection(shadowRay);
			total_light +=
				Graphics::ColorF(light.color()) *
				brdf_color *
				radiance *
				V_PL * V_PL.alpha;
		}

		const float pdf = sample_count / float(light_count);
		return total_light / pdf;
	}
	Graphics::ColorF Kernel::directIllumination(
		const SceneRay& ray,
		const TracingResult& result,
		const SurfaceProperties& surface,
		RNG& rng,
		const RenderConfig& config) const
	{
		const float vS_pdf = BRDF(ray, surface, result.next_direction);
		return 
			directLightSampling(ray, result, surface, vS_pdf, rng, config) + 
			spotLightSampling(ray, result, surface, vS_pdf, rng, config);
	}

	Math::vec3f32 Kernel::spotLightSampleDirection(
		const SpotLight& light,
		const Math::vec3f32& point,
		const Math::vec3f32& vS,
		float& Se,
		RNG& rng) const
	{
		Math::vec3f32 vPL;
		float dPL, vOP_dot_vD, dPQ;
		rayPointCalculation(Ray(point, vS), light.position(), vPL, dPL, vOP_dot_vD, dPQ);

		if (dPQ < light.size() && vOP_dot_vD > 0.0f)
		{	// ray with sample direction would hit the light
			Se = light.emission();
			const float dOQ = sqrtf(dPL * dPL - dPQ * dPQ);
			return vS * fmaxf(dOQ, 1.0e-4f);
		}
		else
		{	// sample random direction on disk
			return sampleDisk(
				rng.unsignedUniform(), rng.unsignedUniform(),
				vPL / dPL, light.size()) + light.position() - point;
		}
	}
	float Kernel::spotLightSolidAngle(const SpotLight& light, const float d) const
	{
		const float A = light.size() * light.size() * std::numbers::pi_v<float>;
		const float d1 = d + 1.0f;
		return A / (d1 * d1);
	}
	float Kernel::spotLightBeamIllumination(const SpotLight& light, const Math::vec3f32& vPL) const
	{
		return float(std::cosf(light.GetBeamAngle()) < Math::vec3f32::Similarity(-vPL, light.direction()));
	}

	Math::vec3f32 Kernel::directLightSampleDirection(
		const DirectLight& light,
		const Math::vec3f32& vS,
		float& Se,
		RNG& rng) const
	{
		const float dot = Math::vec3f32::DotProduct(vS, -light.direction());
		const auto cos_angle = std::cosf(light.angularSize());
		if (dot > cos_angle)
		{	// ray with sample direction would hit the light
			Se = light.emission();
			return vS;
		}
		else
		{	// sample random light direction
			return sampleSphere(
				rng.unsignedUniform(),
				rng.unsignedUniform() *
				0.5f * (1.0f - cos_angle),
				-light.direction());
		}
	}
	float Kernel::directLightSolidAngle(const DirectLight& light) const
	{
		return 2.0f * std::numbers::pi_v<float> *(1.0f - std::cosf(light.angularSize()));
	}
}
