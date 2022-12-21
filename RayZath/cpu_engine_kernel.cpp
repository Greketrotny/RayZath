#include "cpu_engine_kernel.hpp"

#include <numbers>

namespace RayZath::Engine::CPU
{
	void Kernel::setWorld(World& world)
	{
		mp_world = &world;
	}

	Graphics::ColorF Kernel::renderFirstPass(
		const Camera& camera,
		CameraContext& context,
		const Math::vec2ui32 pixel) const
	{
		RZAssertCore(bool(mp_world), "mp_world was nullptr");

		SceneRay ray{};
		generateCameraRay(camera, ray, pixel);

		TracingState tracing_state{Graphics::ColorF(0.0f), 0u};
		auto tracing_result{traceRay(tracing_state, ray)};

		tracing_state.final_color.alpha = 1.0f;
		context.m_image.Value(pixel.x, pixel.y) = tracing_state.final_color;
		return tracing_state.final_color;
	}
	Graphics::ColorF Kernel::renderCumulativePass(
		const Camera& camera,
		CameraContext& context,
		const Math::vec2ui32 pixel) const
	{
		RZAssertCore(bool(mp_world), "mp_world was nullptr");

		return context.m_image.Value(pixel.x, pixel.y);
	}

	TracingResult Kernel::traceRay(TracingState& tracing_state, SceneRay& ray) const
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

		tracing_state.final_color += surface.color;
		return TracingResult{};
	}

	void Kernel::generateCameraRay(const Camera& camera, RangedRay& ray, const Math::vec2ui32& pixel) const
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
		ray.direction = camera.coordSystem().transformForward(ray.direction);
		ray.direction.Normalize();
		ray.origin += camera.position();

		// apply near/far clipping plane
		ray.near_far = camera.nearFar();
	}

	bool Kernel::closestIntersection(SceneRay& ray, SurfaceProperties& surface) const
	{
		const auto& instances = mp_world->container<World::ObjectType::Instance>();
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
	void Kernel::closestIntersection(const Instance& instance, SceneRay& ray, TraversalResult& traversal) const
	{
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
	void Kernel::traverseWorld(const tree_node_t& node, SceneRay& ray, TraversalResult& traversal) const
	{
		if (node.isLeaf())
		{
			for (const auto& object : node.objects())
			{
				if (!object) continue;
				closestIntersection(*object, ray, traversal);
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

	void Kernel::analyzeIntersection(
		const Instance& instance, 
		TraversalResult& traversal, 
		SurfaceProperties& surface) const
	{
		const auto& material = instance.material(traversal.closest_triangle->material_id);
		if (!material) surface.surface_material = &mp_world->defaultMaterial();
		surface.surface_material = material.accessor()->get();
		if (traversal.external) surface.behind_material = surface.surface_material;

		// calculate texture coordinates
		RZAssertCore(instance.mesh(), "If instance had no mesh");
		if (traversal.closest_triangle->texcrds != Mesh::ids_unused)
			surface.texcrd = traversal.closest_triangle->texcrdFromBarycenter(traversal.barycenter, *instance.mesh());

		// calculate mapped normal
		Math::vec3f32 mapped_normal = traversal.closest_triangle->normals != Mesh::ids_unused ?
			traversal.closest_triangle->averageNormal(traversal.barycenter, *instance.mesh()) :
			traversal.closest_triangle->normal;
		if (surface.surface_material->normalMap())
		{
			traversal.closest_triangle->mapNormal(
				Graphics::ColorF(surface.surface_material->normalMap()->fetch(surface.texcrd)),
				mapped_normal,
				*instance.mesh());
		}

		// fill intersection normals
		const float external_factor = static_cast<float>(traversal.external) * 2.0f - 1.0f;
		surface.normal = traversal.closest_triangle->normal * external_factor;
		surface.mapped_normal = mapped_normal * external_factor;

		instance.transformation().transformL2GNoScale(surface.normal);
		surface.normal.Normalize();
		instance.transformation().transformL2GNoScale(surface.mapped_normal);
		surface.mapped_normal.Normalize();
	}


	Graphics::ColorF Kernel::fetchColor(const Material& material, const Texcrd& texcrd) const
	{
		if (const auto& texture = material.texture(); texture)
			return Graphics::ColorF(texture->fetch(texcrd));
		return Graphics::ColorF(material.color());
	}
	float Kernel::fetchMetalness(const Material& material, const Texcrd& texcrd) const
	{
		if (const auto& metalness_map = material.metalnessMap(); metalness_map)
			return metalness_map->fetch(texcrd);
		return material.metalness();
	}
	float Kernel::fetchEmission(const Material& material, const Texcrd& texcrd) const
	{
		if (const auto& emission_map = material.emissionMap(); emission_map)
			return emission_map->fetch(texcrd);
		return material.emission();
	}
	float Kernel::fetchRoughness(const Material& material, const Texcrd& texcrd) const
	{
		if (const auto& roughness_map = material.roughnessMap(); roughness_map)
			return roughness_map->fetch(texcrd);
		return material.roughness();
	}
}
