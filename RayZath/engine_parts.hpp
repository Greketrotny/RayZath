#ifndef ENGINE_PARTS_H
#define ENGINE_PARTS_H

#include <mutex>
#include <map>
#include <vector>
#include <cstdint>

namespace RayZath::Engine
{
	struct ThreadGate
	{
	public:
		enum class State
		{
			Opened,
			Closed
		};
	private:
		State m_state;
		mutable std::mutex m_mtx;
		mutable std::condition_variable m_cv;

	public:
		ThreadGate(State state = State::Closed);

		void open();
		void close();
		void wait();
		void waitAndClose();
		State state() const noexcept;
	};

	struct Timer
	{
	public:
		using duration_t = std::chrono::duration<float, std::milli>;
	private:
		std::chrono::time_point<std::chrono::high_resolution_clock> m_start;

	public:
		Timer();

	public:
		void start();
		duration_t peek();
		duration_t time();
	};

	struct TimeTable
	{
	public:
		using duration_t = RayZath::Engine::Timer::duration_t;
	private:
		float m_avg_factor = 0.05f;
		Timer m_timer, m_cycle_timer;
		struct TimeEntry
		{
			duration_t stage_duration, avg_stage_duration;
			duration_t wait_duration, avg_wait_duration;
		};

	public:
		std::map<std::string_view, size_t> m_entry_map;
		std::vector<std::pair<std::string_view, TimeEntry>> m_entries;

	public:
		explicit operator std::string() const;

		void set(const std::string_view name, duration_t duration);
		void setWaitTime(const std::string_view name, duration_t duration);
		void update(const std::string_view name);
		void updateCycle(const std::string_view name);
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
		uint8_t m_max_depth;
		uint32_t m_rpp;

	public:
		Tracing(
			const uint8_t max_path_depth = 16u,
			const uint32_t rays_per_pixel = 8u);

	public:
		uint8_t maxDepth() const;
		uint32_t rpp() const;

		void maxDepth(const uint8_t max_depth);
		void rpp(const uint32_t rpp);
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
