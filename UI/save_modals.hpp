#pragma once

#include "scene.hpp"

#include <variant>

namespace RayZath::UI::Windows
{
	class SceneSaveModal
	{
	private:
		bool m_opened = true;
		std::array<char, 2048> m_path_buffer{};
		std::optional<std::string> m_fail_message;

		Engine::Saver::SaveOptions m_save_options;

	public:
		void update(Scene& scene);
	};

	class SaveModals
	{
	private:
		std::reference_wrapper<Scene> mr_scene;
		std::variant<
			std::monostate,
			SceneSaveModal
		> m_load_modal;

	public:
		SaveModals(std::reference_wrapper<Scene> scene)
			: mr_scene(std::move(scene))
		{}

		template <typename T>
		void open()
		{
			m_load_modal.emplace<T>();
		}
		void update()
		{
			std::visit([ref = mr_scene](auto& modal)
				{
					if constexpr (!std::is_same_v<std::decay_t<decltype(modal)>, std::monostate>)
						modal.update(ref);
				}, m_load_modal);
		}
	};
}
