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
		std::filesystem::path m_curr_path = "D:\\Users\\Greketrotny\\Documents\\RayZath\\Resources\\img";

		std::vector<std::filesystem::directory_entry> m_directory_content;
		std::unordered_set<size_t> m_selected_items;
		size_t m_last_clicked = 0;

	public:
		FileBrowserModal(std::filesystem::path start_path);

		bool render();
		void directoryContent();

		void loadDirectoryContent();

		std::vector<std::filesystem::path> selectedFiles();
	};
}
