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


	struct LightSampling
	{
	private:
		uint8_t m_point_light, m_spot_light, m_direct_light;

	public:
		LightSampling(
			const uint8_t point_light = 1u, 
			const uint8_t spot_light = 1u, 
			const uint8_t direct_light = 1u);

	public:
		uint8_t GetPointLight() const;
		uint8_t GetSpotLight() const;
		uint8_t GetDirectLight() const;

		void SetPointLight(const uint8_t samples);
		void SetSpotLight(const uint8_t samples);
		void SetDirectLight(const uint8_t samples);
	};

	struct RenderConfig
	{
	private:
		LightSampling m_light_sampling;

	public:
		RenderConfig(LightSampling light_sampling = LightSampling{});

	public:
		LightSampling& GetLightSampling();
		const LightSampling& GetLightSampling() const;
	};
}

#endif // !ENGINE_PARTS_H