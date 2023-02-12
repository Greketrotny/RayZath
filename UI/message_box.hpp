#pragma once

#include <functional>
#include <string>
#include <optional>

namespace RayZath::UI::Windows
{
	template <typename F>
	struct Complete
	{
	private:
		F m_f;
	public:
		Complete(F f)
			: m_f(std::move(f))
		{}
		~Complete() noexcept(std::is_nothrow_invocable_v<F>)
		{
			m_f();
		}
	};

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
}
