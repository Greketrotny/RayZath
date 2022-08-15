#include "engine_parts.hpp"

namespace RayZath::Engine
{
	ThreadGate::ThreadGate(GateState state)
		: m_state(state)
	{}

	void ThreadGate::Open()
	{
		std::lock_guard<std::mutex> lg(m_gate_mutex);
		m_state = GateState::Opened;
		m_cv.notify_all();
	}
	void ThreadGate::Close()
	{
		std::lock_guard<std::mutex> lg(m_gate_mutex);
		m_state = GateState::Closed;
	}
	void ThreadGate::Wait()
	{
		std::unique_lock<std::mutex> lck(m_gate_mutex);
		if (m_state == GateState::Closed)
		{
			m_cv.wait(lck);
		}
	}
	void ThreadGate::WaitAndClose()
	{
		{
			std::unique_lock<std::mutex> lck(m_gate_mutex);
			if (m_state == GateState::Closed)
			{
				m_cv.wait(lck);
			}
		}
		Close();
	}
	ThreadGate::GateState ThreadGate::State() const noexcept
	{
		std::lock_guard<std::mutex> lg(m_gate_mutex);
		return m_state;
	}



	Timer::Timer()
	{
		Start();
	}

	void Timer::Start()
	{
		start = std::chrono::high_resolution_clock::now();
	}
	float Timer::PeekTime()
	{
		auto stop = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> duration = stop - start;
		return duration.count();
	}
	float Timer::GetTime()
	{
		auto stop = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> duration = stop - start;
		start = stop;
		return duration.count();
	}


	// ~~~~~~~~ LightSampling ~~~~~~~~
	LightSampling::LightSampling(
		const uint8_t spot_light,
		const uint8_t direct_light)
		: m_spot_light(spot_light)
		, m_direct_light(direct_light) {}

	uint8_t LightSampling::GetSpotLight() const { return m_spot_light; }
	uint8_t LightSampling::GetDirectLight() const { return m_direct_light; }

	void LightSampling::SetSpotLight(const uint8_t samples)
	{
		m_spot_light = samples;
	}
	void LightSampling::SetDirectLight(const uint8_t samples)
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

	uint8_t Tracing::GetMaxDepth() const
	{
		return m_max_depth;
	}
	uint8_t Tracing::GetRPP() const
	{
		return m_rpp;
	}

	void Tracing::SetMaxDepth(const uint8_t max_depth)
	{
		m_max_depth = max_depth;
	}
	void Tracing::SetRPP(const uint8_t rpp)
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

	LightSampling& RenderConfig::GetLightSampling()
	{
		return m_light_sampling;
	}
	const LightSampling& RenderConfig::GetLightSampling() const
	{
		return m_light_sampling;
	}
	Tracing& RenderConfig::GetTracing()
	{
		return m_tracing;
	}
	const Tracing& RenderConfig::GetTracing() const
	{
		return m_tracing;
	}
}