#pragma once

#include <filesystem>
#include <array>
#include <unordered_set>

namespace RayZath::UI::Windows
{
	class FileBrowserModal
	{
	private:
		bool m_opened = true;

		std::array<char, 2048> m_path_buff{};

		std::list<std::filesystem::path> m_path_history;
		decltype(m_path_history)::iterator m_curr_path;

		std::vector<std::filesystem::directory_entry> m_directory_content;
		std::unordered_set<size_t> m_selected_items;
		size_t m_last_clicked = 0;

		std::string m_error_string;
	public:
		FileBrowserModal(std::filesystem::path start_path);

		bool render();
		std::vector<std::filesystem::path> selectedFiles();
	private:
		void renderNavButtons();
		void renderPathBar();
		void renderDirectoryContent();

		void loadDirectoryContent();
		void setCurrentPath(std::filesystem::path new_path);
	};
}
