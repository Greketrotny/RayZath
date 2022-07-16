#pragma once

#include "scene.hpp"

#include <variant>

namespace RayZath::UI::Windows
{
	class SceneExplorer;

	using namespace std::string_view_literals;

	template <Engine::World::ObjectType T>
	class LoadModal
	{
	protected:
		bool m_opened = true;
		std::reference_wrapper<SceneExplorer> mr_explorer;
		std::array<char, 2048> m_path_buffer{};
		std::optional<std::string> m_fail_message;

		static constexpr std::array ms_filter_modes = {
			std::make_pair(Engine::Texture::FilterMode::Linear, "linear"sv),
			std::make_pair(Engine::Texture::FilterMode::Point, "point"sv) };
		static constexpr std::array ms_address_modes = {
			std::make_pair(Engine::Texture::AddressMode::Wrap, "wrap"sv),
			std::make_pair(Engine::Texture::AddressMode::Clamp, "clamp"sv),
			std::make_pair(Engine::Texture::AddressMode::Mirror, "mirror"sv),
			std::make_pair(Engine::Texture::AddressMode::Border, "border"sv) };
		int m_filter_mode_idx = 0, m_addres_mode_idx = 0;

	public:
		LoadModal(std::reference_wrapper<SceneExplorer> explorer)
			: mr_explorer(std::move(explorer))
		{}
		void update(Scene& scene);
	};

	class LoadModals
	{
	private:
		std::reference_wrapper<Scene> mr_scene;
		std::variant<
			std::monostate,
			//LoadModal<Engine::World::ObjectType::Texture>,
			//LoadModal<Engine::World::ObjectType::NormalMap>,
			//LoadModal<Engine::World::ObjectType::MetalnessMap>,
			//LoadModal<Engine::World::ObjectType::RoughnessMap>,
			LoadModal<Engine::World::ObjectType::Texture>> m_load_modal;

	public:
		LoadModals(std::reference_wrapper<Scene> scene)
			: mr_scene(std::move(scene))
		{}

		template <typename T>
		void open(std::reference_wrapper<SceneExplorer> explorer)
		{
			m_load_modal.emplace<T>(std::move(explorer));
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
