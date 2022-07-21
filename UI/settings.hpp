#pragma once

#include "rayzath.h"

namespace RayZath::UI::Windows
{
	class Settings
	{
	private:
		bool m_opened = false;

	public:
		void open();
		bool isOpened() const;

		void update();
	};
}
