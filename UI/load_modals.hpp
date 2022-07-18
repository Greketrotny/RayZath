#pragma once

#include "scene.hpp"
#include "explorer.hpp"

#include <variant>

namespace RayZath::UI::Windows
{
	class SceneExplorer;

	using namespace std::string_view_literals;

	template <Engine::World::ObjectType T>
	class LoadMapModalBase
	{
	protected:
		bool m_opened = true;
		std::reference_wrapper<SceneExplorer> mr_explorer;
		std::array<char, 2048> m_path_buffer{};
		std::optional<std::string> m_fail_message;

		template <Engine::World::ObjectType U>
		using map_t = Utils::static_dictionary::vt_translate<U>::template with<
			Utils::static_dictionary::vt_translation<Engine::World::ObjectType::Texture, RZ::Texture>,
			Utils::static_dictionary::vt_translation<Engine::World::ObjectType::NormalMap, RZ::NormalMap>,
			Utils::static_dictionary::vt_translation<Engine::World::ObjectType::MetalnessMap, RZ::MetalnessMap>,
			Utils::static_dictionary::vt_translation<Engine::World::ObjectType::RoughnessMap, RZ::RoughnessMap>,
			Utils::static_dictionary::vt_translation<Engine::World::ObjectType::EmissionMap, RZ::EmissionMap>>::template value;

		static constexpr std::array ms_filter_modes = {
			std::make_pair(map_t<T>::FilterMode::Linear, "linear"sv),
			std::make_pair(map_t<T>::FilterMode::Point, "point"sv) };
		static constexpr std::array ms_address_modes = {
			std::make_pair(map_t<T>::AddressMode::Wrap, "wrap"sv),
			std::make_pair(map_t<T>::AddressMode::Clamp, "clamp"sv),
			std::make_pair(map_t<T>::AddressMode::Mirror, "mirror"sv),
			std::make_pair(map_t<T>::AddressMode::Border, "border"sv) };
		int m_filter_mode_idx = 0, m_addres_mode_idx = 0;

	public:
		LoadMapModalBase(std::reference_wrapper<SceneExplorer> explorer)
			: mr_explorer(std::move(explorer))
		{}
	};

	template <Engine::World::ObjectType T>
	class LoadModal;

	template <Engine::World::ObjectType T> requires MapObjectType<T>
	class LoadModal<T> : public LoadMapModalBase<T>
	{
	private:
		using base_t = LoadMapModalBase<T>;
		using base_t::mr_explorer;

	public:
		LoadModal(std::reference_wrapper<SceneExplorer> explorer)
			: base_t(std::move(explorer))
		{}
		void update(Scene& scene);
	};
	template<>
	class LoadModal<Engine::World::ObjectType::EmissionMap>
		: public LoadMapModalBase<Engine::World::ObjectType::EmissionMap>
	{
	protected:
		using base_t = LoadMapModalBase<Engine::World::ObjectType::EmissionMap>;
		using base_t::mr_explorer;
		float m_emission_factor = 1.0f;

	public:
		LoadModal(std::reference_wrapper<SceneExplorer> explorer)
			: base_t(std::move(explorer))
		{}

		void update(Scene& scene);
	};

	template<>
	class LoadModal<Engine::World::ObjectType::Material>
	{
	protected:
		bool m_opened = true;
		std::reference_wrapper<SceneExplorer> mr_explorer;
		std::array<char, 2048> m_path_buffer{};
		std::optional<std::string> m_fail_message;

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
			LoadModal<Engine::World::ObjectType::Texture>,
			LoadModal<Engine::World::ObjectType::NormalMap>,
			LoadModal<Engine::World::ObjectType::MetalnessMap>,
			LoadModal<Engine::World::ObjectType::RoughnessMap>,
			LoadModal<Engine::World::ObjectType::EmissionMap>,
			LoadModal<Engine::World::ObjectType::Material>
		> m_load_modal;

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
