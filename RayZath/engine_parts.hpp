#ifndef ENGINE_PARTS_H
#define ENGINE_PARTS_H

#include <mutex>

namespace RayZath::Engine
{
	struct ThreadGate
	{
	public:
		enum class GateState
		{
			Opened,
			Closed
		};
	private:
		GateState m_state;
		mutable std::mutex m_gate_mutex;
		std::condition_variable m_cv;


	public:
		ThreadGate(GateState state = GateState::Closed);


	public:
		void Open();
		void Close();
		void Wait();
		void WaitAndClose();
		GateState State() const noexcept;
	};

	struct Timer
	{
	private:
		std::chrono::time_point<std::chrono::high_resolution_clock> start;

	public:
		Timer();

	public:
		void Start();
		float PeekTime();
		float GetTime();
	};


	struct LightSampling
	{
	private:
		uint8_t m_spot_light, m_direct_light;

	public:
		LightSampling(
			const uint8_t spot_light = 1u, 
			const uint8_t direct_light = 1u);

	public:
		uint8_t GetSpotLight() const;
		uint8_t GetDirectLight() const;

		void SetSpotLight(const uint8_t samples);
		void SetDirectLight(const uint8_t samples);
	};
	struct Tracing
	{
	private:
		uint8_t m_max_depth, m_rpp;

	public:
		Tracing(
			const uint8_t max_path_depth = 16u,
			const uint8_t rays_per_pixel = 8u);

	public:
		uint8_t GetMaxDepth() const;
		uint8_t GetRPP() const;

		void SetMaxDepth(const uint8_t max_depth);
		void SetRPP(const uint8_t rpp);
	};

	struct RenderConfig
	{
	private:
		LightSampling m_light_sampling;
		Tracing m_tracing;

	public:
		RenderConfig(
			LightSampling light_sampling = LightSampling{},
			Tracing tracing = Tracing{});

	public:
		LightSampling& GetLightSampling();
		const LightSampling& GetLightSampling() const;
		Tracing& GetTracing();
		const Tracing& GetTracing() const;
	};
}

#endif // !ENGINE_PARTS_H