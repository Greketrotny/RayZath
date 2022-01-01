#include "engine_parts.h"

namespace RayZath::Engine
{
	// ~~~~~~~~ ThreadGate ~~~~~~~~
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
		return m_state;
	}


	// ~~~~~~~~ Timer ~~~~~~~~
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


	// ~~~~~~~~ LightSamples ~~~~~~~~
	LightSampling::LightSampling(
		const uint8_t point_light,
		const uint8_t spot_light,
		const uint8_t direct_light)
		: m_point_light(point_light)
		, m_spot_light(spot_light)
		, m_direct_light(direct_light) {}

	uint8_t LightSampling::GetPointLight() const { return m_point_light; }
	uint8_t LightSampling::GetSpotLight() const { return m_spot_light; }
	uint8_t LightSampling::GetDirectLight() const { return m_direct_light; }

	void LightSampling::SetPointLight(const uint8_t samples)
	{
		m_point_light = samples;
	}
	void LightSampling::SetSpotLight(const uint8_t samples)
	{
		m_spot_light = samples;
	}
	void LightSampling::SetDirectLight(const uint8_t samples)
	{
		m_direct_light = samples;
	}


	// ~~~~~~~~ RenderConfig ~~~~~~~~
	RenderConfig::RenderConfig(LightSampling light_sampling)
		: m_light_sampling(light_sampling)
	{}

	LightSampling& RenderConfig::GetLightSampling()
	{
		return m_light_sampling;
	}
	const LightSampling& RenderConfig::GetLightSampling() const
	{
		return m_light_sampling;
	}
}