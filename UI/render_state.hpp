#pragma once

#include "rayzath.hpp"

namespace RayZath::UI::Windows
{
	class RenderState
	{
	private:
		bool m_opened = false;

	public:
		void open();
		bool isOpened() const;

		void update();
	};
}
