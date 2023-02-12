#pragma once

#include <filesystem>
#include <array>
#include <unordered_set>
#include <functional>

namespace RayZath::UI::Windows
{
	class MessageBox
	{
	public:
		using option_t = std::optional<std::string>;
		using callback_t = std::function<void(option_t)>;
	private:
		bool m_opened = true;
		std::string m_message;
		std::vector<std::string> m_options;
		callback_t m_callback;

	public:
		MessageBox();
		MessageBox(std::string message, std::vector<std::string> options, callback_t callback = {});

		option_t render();
	};

	class FileBrowserModal
	{
	public:
		enum class Mode
		{
			Open,
			Save
		};
	private:
		bool m_opened = true;
		Mode m_mode = Mode::Open;

		std::array<char, 2048> m_path_buff{}, m_file_buff{}, m_new_folder_buff{};

		std::list<std::filesystem::path> m_path_history;
		decltype(m_path_history)::iterator m_curr_path;

		std::vector<std::filesystem::directory_entry> m_directory_content;
		std::unordered_set<size_t> m_selected_items;
		size_t m_last_clicked = 0;
		bool m_adding_new_folder = false;

		MessageBox m_message_box;
	public:
		FileBrowserModal(std::filesystem::path start_path, Mode mode);

		bool render();
		std::vector<std::filesystem::path> selectedFiles();
	private:
		void renderNavButtons();
		void renderPathBar();
		void renderDirectoryContent();
		void renderNewFolderInput();
		bool renderBottomBar();

		void loadDirectoryContent();
		void setCurrentPath(std::filesystem::path new_path);
	};
}
