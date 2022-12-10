#include "engine_parts.hpp"

#include <sstream>
#include <iomanip>
#include <algorithm>

namespace RayZath::Engine
{
	ThreadGate::ThreadGate(State state)
		: m_state(state)
	{}

	void ThreadGate::open()
	{
		std::scoped_lock lck(m_mtx);
		m_state = State::Opened;
		m_cv.notify_all();
	}
	void ThreadGate::close()
	{
		std::scoped_lock lck(m_mtx);
		m_state = State::Closed;
	}
	void ThreadGate::wait()
	{
		std::unique_lock<std::mutex> lck(m_mtx);
		m_cv.wait(lck, [this]() { return m_state == State::Opened; });
	}
	void ThreadGate::waitAndClose()
	{
		std::unique_lock<std::mutex> lck(m_mtx);
		m_cv.wait(lck, [this]() { return m_state == State::Opened; });
		m_state = State::Closed;
	}
	ThreadGate::State ThreadGate::state() const noexcept
	{
		return m_state;
	}


	Timer::Timer()
	{
		start();
	}

	void Timer::start()
	{
		m_start = std::chrono::high_resolution_clock::now();
	}
	Timer::duration_t Timer::peek()
	{
		return std::chrono::high_resolution_clock::now() - m_start;
	}
	Timer::duration_t Timer::time()
	{
		auto stop = std::chrono::high_resolution_clock::now();
		duration_t duration = stop - m_start;
		m_start = stop;
		return duration;
	}

	TimeTable::operator std::string() const
	{
		auto it = std::max_element(m_entry_map.begin(), m_entry_map.end(), [](const auto& left, const auto& right)
			{
				return left.first.size() < right.first.size();
			});
		if (it == m_entry_map.end()) return "empty";
		const size_t width = it->first.length();

		std::stringstream ss;
		for (const auto& [name, entry] : m_entries)
		{
			ss << std::setfill(' ') << std::setw(width) << name << ": ";
			ss << std::fixed << std::setprecision(3) << std::setfill(' ');
			ss << std::setw(5) << entry.stage_duration.count();
			ss << " (" << std::setw(5) << entry.avg_stage_duration.count() << ")";
			ss << ", " << std::setw(5) << entry.wait_duration.count();
			ss << " (" << std::setw(5) << entry.avg_wait_duration.count() << ")";
			ss << std::endl;
		}
		return ss.str();
	}
	void TimeTable::set(const std::string_view name, TimeTable::duration_t duration)
	{
		auto entry_it = m_entry_map.find(name);
		if (entry_it == m_entry_map.end())
		{
			m_entries.push_back({name, {}});
			entry_it = m_entry_map.insert({name, m_entries.size() - 1}).first;
		}
		auto& entry = m_entries[entry_it->second].second;
		entry.stage_duration = duration;
		entry.avg_stage_duration += (duration - entry.avg_stage_duration) * m_avg_factor;
	}
	void TimeTable::setWaitTime(const std::string_view name, TimeTable::duration_t duration)
	{
		auto entry_it = m_entry_map.find(name);
		if (entry_it == m_entry_map.end())
		{
			m_entries.push_back({name, {}});
			entry_it = m_entry_map.insert({name, m_entries.size() - 1}).first;
		}
		auto& entry = m_entries[entry_it->second].second;
		entry.wait_duration = duration;
		entry.avg_wait_duration += (duration - entry.avg_wait_duration) * m_avg_factor;
		m_timer.start();
	}
	void TimeTable::update(const std::string_view name)
	{
		set(name, m_timer.time());
	}
	void TimeTable::updateCycle(const std::string_view name)
	{
		set(name, m_cycle_timer.time());
	}


	// ~~~~~~~~ LightSampling ~~~~~~~~
	LightSampling::LightSampling(
		const uint8_t spot_light,
		const uint8_t direct_light)
		: m_spot_light(spot_light)
		, m_direct_light(direct_light) {}

	uint8_t LightSampling::spotLight() const { return m_spot_light; }
	uint8_t LightSampling::directLight() const { return m_direct_light; }

	void LightSampling::spotLight(const uint8_t samples)
	{
		m_spot_light = samples;
	}
	void LightSampling::directLight(const uint8_t samples)
	{
		m_direct_light = samples;
	}
	// ~~~~~~~~ Tracing ~~~~~~~~
	Tracing::Tracing(
		const uint8_t max_path_depth,
		const uint8_t rays_per_pixel)
		: m_max_depth(max_path_depth)
		, m_rpp(rays_per_pixel)
	{}

	uint8_t Tracing::maxDepth() const
	{
		return m_max_depth;
	}
	uint8_t Tracing::rpp() const
	{
		return m_rpp;
	}

	void Tracing::maxDepth(const uint8_t max_depth)
	{
		m_max_depth = max_depth;
	}
	void Tracing::rpp(const uint8_t rpp)
	{
		m_rpp = rpp;
	}


	// ~~~~~~~~ RenderConfig ~~~~~~~~
	RenderConfig::RenderConfig(
		LightSampling light_sampling,
		Tracing tracing)
		: m_light_sampling(std::move(light_sampling))
		, m_tracing(std::move(tracing))
	{}

	LightSampling& RenderConfig::lightSampling()
	{
		return m_light_sampling;
	}
	const LightSampling& RenderConfig::lightSampling() const
	{
		return m_light_sampling;
	}
	Tracing& RenderConfig::tracing()
	{
		return m_tracing;
	}
	const Tracing& RenderConfig::tracing() const
	{
		return m_tracing;
	}
}