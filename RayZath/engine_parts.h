#ifndef ENGINE_PARTS_H
#define ENGINE_PARTS_H

#include <mutex>

namespace RayZath
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
		std::mutex m_gate_mutex;
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
}

#endif // !ENGINE_PARTS_H