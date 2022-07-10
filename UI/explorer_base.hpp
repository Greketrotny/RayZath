#pragma once

#include <array>
#include <string>

namespace RayZath::UI::Windows
{
	class ExplorerSelectable
	{
	public:
		struct Action
		{
			bool selected = false;
			bool double_clicked = false;
			bool right_clicked = false;
		};

		Action drawSelectable(const std::string& caption, const bool selected);
	};
	class ExplorerEditable : private ExplorerSelectable
	{
	private:
		std::array<char, 256> m_name_buffer;

	public:
		struct Action : ExplorerSelectable::Action
		{
			bool name_edited = false;

			Action() = default;
			Action(const ExplorerSelectable::Action& other_base)
				: ExplorerSelectable::Action(other_base)
			{}
		};

		Action drawEditable(const std::string& caption, const bool selected, const bool edited);
		void setNameToEdit(const std::string& name);
		std::string getEditedName();
	};
}
