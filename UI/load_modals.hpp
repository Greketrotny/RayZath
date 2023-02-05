#pragma once

#include "scene.hpp"
#include "explorer.hpp"
#include "file_browser.hpp"

#include <variant>

namespace RayZath::UI::Windows
{
	class SceneExplorer;

	using namespace std::string_view_literals;

	template <Engine::ObjectType T>
	class LoadMapModalBase
	{
	protected:
		bool m_opened = true;
		std::reference_wrapper<SceneExplorer> mr_explorer;
		std::vector<std::filesystem::path> m_files_to_load;
		std::array<char, 2048> m_path_buffer{};
		std::optional<std::string> m_fail_message;

		std::optional<FileBrowserModal> m_file_browser;
		

		template <Engine::ObjectType U>
		using map_t = typename Utils::static_dictionary::vt_translate<U>::template with<
			Utils::static_dictionary::vt_translation<Engine::ObjectType::Texture, RZ::Texture>,
			Utils::static_dictionary::vt_translation<Engine::ObjectType::NormalMap, RZ::NormalMap>,
			Utils::static_dictionary::vt_translation<Engine::ObjectType::MetalnessMap, RZ::MetalnessMap>,
			Utils::static_dictionary::vt_translation<Engine::ObjectType::RoughnessMap, RZ::RoughnessMap>,
			Utils::static_dictionary::vt_translation<Engine::ObjectType::EmissionMap, RZ::EmissionMap>>::value;

		static constexpr std::array ms_filter_modes = {
			std::make_pair(map_t<T>::FilterMode::Linear, "linear"sv),
			std::make_pair(map_t<T>::FilterMode::Point, "point"sv) };
		static constexpr std::array ms_address_modes = {
			std::make_pair(map_t<T>::AddressMode::Wrap, "wrap"sv),
			std::make_pair(map_t<T>::AddressMode::Clamp, "clamp"sv),
			std::make_pair(map_t<T>::AddressMode::Mirror, "mirror"sv),
			std::make_pair(map_t<T>::AddressMode::Border, "border"sv) };
		size_t m_filter_mode_idx = 0, m_addres_mode_idx = 0;

	public:
		LoadMapModalBase(std::reference_wrapper<SceneExplorer> explorer)
			: mr_explorer(std::move(explorer))
		{}
	};

	template <Engine::ObjectType T>
	class LoadModal;

	template <Engine::ObjectType T> requires MapObjectType<T>
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
	class LoadModal<Engine::ObjectType::Texture>
		: public LoadMapModalBase<Engine::ObjectType::Texture>
	{
	protected:
		using base_t = LoadMapModalBase<Engine::ObjectType::Texture>;
		using base_t::mr_explorer;
		float m_emission_factor = 1.0f;
		bool m_is_hdr = false;

	public:
		LoadModal(std::reference_wrapper<SceneExplorer> explorer)
			: base_t(std::move(explorer))
		{}

		void update(Scene& scene);
		void doLoad(Scene& scene, const std::filesystem::path& file);
	};
	template<>
	class LoadModal<Engine::ObjectType::NormalMap>
		: public LoadMapModalBase<Engine::ObjectType::NormalMap>
	{
	protected:
		using base_t = LoadMapModalBase<Engine::ObjectType::NormalMap>;
		using base_t::mr_explorer;
		bool m_flip_y_axis = false;

	public:
		LoadModal(std::reference_wrapper<SceneExplorer> explorer)
			: base_t(std::move(explorer))
		{}

		void update(Scene& scene);
	};
	template<>
	class LoadModal<Engine::ObjectType::EmissionMap>
		: public LoadMapModalBase<Engine::ObjectType::EmissionMap>
	{
	protected:
		using base_t = LoadMapModalBase<Engine::ObjectType::EmissionMap>;
		using base_t::mr_explorer;
		float m_emission_factor = 1.0f;

	public:
		LoadModal(std::reference_wrapper<SceneExplorer> explorer)
			: base_t(std::move(explorer))
		{}

		void update(Scene& scene);
	};

	template<>
	class LoadModal<Engine::ObjectType::Material>
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

	template<>
	class LoadModal<Engine::ObjectType::Mesh>
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
	class SceneLoadModal
	{
	protected:
		bool m_opened = true;
		std::reference_wrapper<SceneExplorer> mr_explorer;
		std::array<char, 2048> m_path_buffer{};
		std::optional<std::string> m_fail_message;

	public:
		SceneLoadModal(std::reference_wrapper<SceneExplorer> explorer)
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
			LoadModal<Engine::ObjectType::Texture>,
			LoadModal<Engine::ObjectType::NormalMap>,
			LoadModal<Engine::ObjectType::MetalnessMap>,
			LoadModal<Engine::ObjectType::RoughnessMap>,
			LoadModal<Engine::ObjectType::EmissionMap>,
			LoadModal<Engine::ObjectType::Material>,
			LoadModal<Engine::ObjectType::Mesh>,
			SceneLoadModal
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
