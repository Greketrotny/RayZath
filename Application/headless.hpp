#pragma once

#include <filesystem>

namespace RayZath::Headless
{
	class Headless
	{
	private:
		float m_load_time = 0.1f, m_floaty_rpp = 1.0f;

	public:
		static Headless& instance();

		int run(
			std::filesystem::path scene_path,
			std::filesystem::path report_path,
			std::filesystem::path config_path);

	private:
		void render();
	};
}

