#pragma once

#include "rayzath.hpp"

namespace RayZath::UI::Windows
{
	class RenderState
	{
	private:
		bool m_opened =
			#ifdef NDEBUG
			false;
		#else
			true;
		#endif // NDEBUG


	public:
		void open();
		bool isOpened() const;

		void update();
	};
}
