#include "engine_parts.h"

namespace RayZath
{
	// ~~~~~~~~ [STRUCT] ThreadGate ~~~~~~~~
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
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	// ~~~~~~~~ [STRUCT] Timer ~~~~~~~~
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
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}