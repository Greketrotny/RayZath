#pragma once

#include <filesystem>
#include <array>

namespace RayZath::UI::Windows
{
	class FileBrowserModal
	{
	private:
		bool m_opened = true;

		std::array<char, 2048> m_path_buff{};
		std::filesystem::path m_curr_path = "D:\\Users\\Greketrotny\\Documents\\RayZath\\Resources\\img";
		bool m_as_input = false;
		std::vector<std::filesystem::directory_entry> m_directory_content;

	public:
		std::filesystem::path m_selected_file;

		FileBrowserModal(std::filesystem::path start_path);

		bool render();
		void directoryContent();

		void loadDirectoryContent();

	};
}
