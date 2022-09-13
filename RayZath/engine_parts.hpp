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
		void open();
		void close();
		void wait();
		void waitAndClose();
		GateState state() const noexcept;
	};

	struct Timer
	{
	private:
		std::chrono::time_point<std::chrono::high_resolution_clock> m_start;

	public:
		Timer();

	public:
		void start();
		float peekTime();
		float time();
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
		uint8_t spotLight() const;
		uint8_t directLight() const;

		void spotLight(const uint8_t samples);
		void directLight(const uint8_t samples);
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
		uint8_t maxDepth() const;
		uint8_t rpp() const;

		void maxDepth(const uint8_t max_depth);
		void rpp(const uint8_t rpp);
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
		LightSampling& lightSampling();
		const LightSampling& lightSampling() const;
		Tracing& tracing();
		const Tracing& tracing() const;
	};
}

#endif // !ENGINE_PARTS_H