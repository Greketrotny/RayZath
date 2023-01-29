#ifndef CPU_ENGINE_KERNEL_HPP
#define CPU_ENGINE_KERNEL_HPP

#include "world.hpp"

#include "cpu_render_utils.hpp"
#include "engine_parts.hpp"

#include <type_traits>

namespace RayZath::Engine::CPU
{
	struct TracingState
	{
		Graphics::ColorF final_color;
		uint8_t path_depth;
		static constexpr uint8_t sm_path_limit = std::numeric_limits<decltype(path_depth)>::max();

		TracingState(const Graphics::ColorF color, const uint8_t depth)
			: final_color(color)
			, path_depth(depth)
		{}

		void endPath()
		{
			path_depth = sm_path_limit;
		}
	};
	struct CameraContext
	{
		Graphics::Buffer2D<Graphics::ColorF> m_image;

		Graphics::Buffer2D<uint8_t> m_path_depth;

		Graphics::Buffer2D<Math::vec3f32> m_ray_origin;
		Graphics::Buffer2D<Math::vec3f32> m_ray_direction;
		Graphics::Buffer2D<const Material*> m_ray_material;
		Graphics::Buffer2D<Graphics::ColorF> m_ray_color;

		bool m_update_flag = true;
		uint64_t m_traced_rays = 0;
		std::atomic<uint32_t> m_block_id = 0;

		CameraContext(Math::vec2ui32 resolution = Math::vec2ui32(1, 1));

		void resize(const Math::vec2ui32 resolution);
		void reset(const Math::vec2ui32 resolution);

		void setRay(const Math::vec2ui32 pixel, const SceneRay& ray);
		SceneRay getRay(const Math::vec2ui32 pixel);
	};

	class Kernel
	{
	private:
		const World* mp_world = nullptr;

		using tree_node_t = std::decay_t<decltype(mp_world->container<ObjectType::Instance>())>::tree_node_t;

	public:
		void setWorld(World& world);
		Graphics::ColorF renderFirstPass(
			Camera& camera, 
			CameraContext& context,
			const Math::vec2ui32 pixel,
			RNG& rng,
			const RenderConfig& config) const;
		Graphics::ColorF renderCumulativePass(
			const Camera& camera,
			CameraContext& context,
			const Math::vec2ui32 pixel,
			RNG& rng,
			const RenderConfig& config) const;
		void rayCast(Camera& camera) const;

	private:
		void generateSimpleRay(
			const Camera& camera, 
			RangedRay& ray, 
			const Math::vec2ui32& pixel) const;
		void generateAntialiasedRay(
			const Camera& camera, 
			RangedRay& ray, 
			const Math::vec2ui32& pixel,
			RNG& rng) const;

		TracingResult traceRay(
			TracingState& tracing_state, 
			SceneRay& ray, 
			RNG& rng,
			const RenderConfig& config) const;
		void traverseWorld(const tree_node_t& node, RangedRay& ray, TraversalResult& traversal) const;

		bool closestIntersection(RangedRay& ray, SurfaceProperties& surface) const;
		void closestIntersection(const Handle<Instance>& instance, RangedRay& ray, TraversalResult& traversal) const;
		void closestIntersection(const Mesh& mesh, RangedRay& ray, TraversalResult& traversal) const;

		void analyzeIntersection(
			const Instance& instance, 
			TraversalResult& traversal, 
			SurfaceProperties& surface) const;

		Graphics::ColorF anyIntersection(const RangedRay& ray) const;
		Graphics::ColorF anyIntersection(const Instance& instance, const RangedRay& ray) const;
		Graphics::ColorF anyIntersection(const Mesh& mesh, const RangedRay& ray) const;

		std::tuple<Handle<Instance>, Handle<Material>> worldRayCast(RangedRay& ray) const;


		// ~~~~~~~~ material functions ~~~~~~~~
		// texture/map fetching
		Graphics::ColorF fetchColor(const Material& material, const Texcrd& texcrd) const;
		float fetchMetalness(const Material& material, const Texcrd& texcrd) const;
		float fetchEmission(const Material& material, const Texcrd& texcrd) const;
		float fetchRoughness(const Material& material, const Texcrd& texcrd) const;

		// attenuation
		bool applyScattering(const Material& material, SceneRay& ray, SurfaceProperties& surface, RNG& rng) const;

		// bidirectional reflection distribution function
		float BRDF(const RangedRay& ray, const SurfaceProperties& surface, const Math::vec3f32& vPL) const;
		Graphics::ColorF BRDFColor(const SurfaceProperties& surface) const;

		// normal distribution function
		float NDF(const Math::vec3f32 vN, const Math::vec3f32 vH, const float roughness) const;
		float attenuation(const float cos_angle, const float roughness) const;

		// ray generation
		Math::vec3f32 sampleDirection(SceneRay& ray, SurfaceProperties& surface, RNG& rng) const;
		Math::vec3f32 sampleDiffuseDirection(SurfaceProperties& surface, RNG& rng) const;
		Math::vec3f32 sampleGlossyDirection(SceneRay& ray, SurfaceProperties& surface, RNG& rng) const;
		Math::vec3f32 sampleTransmissionDirection(SceneRay& ray, SurfaceProperties& surface, RNG& rng) const;
		Math::vec3f32 sampleScatteringDirection(SceneRay& ray, SurfaceProperties& surface, RNG& rng) const;

		// ~~~~~~~~ direct sampling ~~~~~~~~
		Graphics::ColorF spotLightSampling(
			const SceneRay& ray,
			const TracingResult& result,
			const SurfaceProperties& surface,
			const float vS_pdf,
			RNG& rng,
			const RenderConfig& config) const;
		Graphics::ColorF directLightSampling(
			const SceneRay& ray,
			const TracingResult& result,
			const SurfaceProperties& surface,
			const float vS_pdf,
			RNG& rng,
			const RenderConfig& config) const;
		Graphics::ColorF directIllumination(
			const SceneRay& ray,
			const TracingResult& result,
			const SurfaceProperties& surface,
			RNG& rng,
			const RenderConfig& config) const;

		Math::vec3f32 spotLightSampleDirection(
			const SpotLight& light,
			const Math::vec3f32& point,
			const Math::vec3f32& vS,
			float& Se,
			RNG& rng) const;
		float spotLightSolidAngle(const SpotLight& light, const float d) const;
		float spotLightBeamIllumination(const SpotLight& light, const Math::vec3f32& vPL) const;

		Math::vec3f32 directLightSampleDirection(
			const DirectLight& light,
			const Math::vec3f32& vS,
			float& Se,
			RNG& rng) const;
		float directLightSolidAngle(const DirectLight& light) const;
	};
}

#endif 
