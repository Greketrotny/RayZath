#include "rendering.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include "glfw.hpp"
#include "vulkan.hpp"

#include "rayzath.hpp"

#include <stdexcept>
#include <iostream>
#include <string>
#include <functional>
#include <chrono>

namespace RayZath::UI::Rendering
{
	static void check_vk_result(VkResult err)
	{
		if (err != 0)
			throw std::exception((std::string("[vulkan] Error: VkResult = ") + std::to_string(err)).c_str());
	}

	Module::Module()
		: m_vulkan(m_glfw)
	{
		// Setup Dear ImGui context
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		setImguiStyle();

		// setup backend
		ImGui_ImplGlfw_InitForVulkan(m_glfw.window(), true);
		ImGui_ImplVulkan_InitInfo init_info = {};
		init_info.Instance = Vulkan::Instance::get().vulkanInstance();
		init_info.PhysicalDevice = Vulkan::Instance::get().physicalDevice();
		init_info.Device = Vulkan::Instance::get().logicalDevice();
		init_info.QueueFamily = Vulkan::Instance::get().queueFamilyIdx();
		init_info.Queue = Vulkan::Instance::get().queue();
		init_info.PipelineCache = VK_NULL_HANDLE;
		init_info.DescriptorPool = Vulkan::Instance::get().descriptorPool();
		init_info.Subpass = 0;
		init_info.MinImageCount = m_vulkan.window().m_min_image_count;
		init_info.ImageCount = init_info.MinImageCount;
		init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
		init_info.Allocator = Vulkan::Instance::get().allocator();
		init_info.CheckVkResultFn = check_vk_result;
		ImGui_ImplVulkan_Init(&init_info, m_vulkan.window().renderPass());

		// Load Fonts
		// - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
		// - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
		// - If the file cannot be loaded, the function will return NULL. Please handle those errors in your Rendering (e.g. use an assertion, or display an error and quit).
		// - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
		// - Read 'docs/FONTS.md' for more instructions and details.
		//io.Fonts->AddFontDefault();
		//io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
		//io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
		//io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
		//io.Fonts->AddFontFromFileTTF("../../misc/fonts/ProggyTiny.ttf", 10.0f);
		//ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());
		//IM_ASSERT(font != NULL);

		// Upload Fonts
		{
			// Use any command queue
			VkCommandPool command_pool = m_vulkan.m_window.currentFrame().m_command_pool;
			VkCommandBuffer command_buffer = m_vulkan.m_window.currentFrame().commandBuffer();

			Vulkan::check(vkResetCommandPool(Vulkan::Instance::get().logicalDevice(), command_pool, 0));
			VkCommandBufferBeginInfo begin_info{};
			begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			Vulkan::check(vkBeginCommandBuffer(command_buffer, &begin_info));

			ImGui_ImplVulkan_CreateFontsTexture(command_buffer);

			VkSubmitInfo end_info = {};
			end_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			end_info.commandBufferCount = 1;
			end_info.pCommandBuffers = &command_buffer;
			Vulkan::check(vkEndCommandBuffer(command_buffer));
			Vulkan::check(vkQueueSubmit(Vulkan::Instance::get().queue(), 1, &end_info, VK_NULL_HANDLE));

			Vulkan::check(vkDeviceWaitIdle(Vulkan::Instance::get().logicalDevice()));
			ImGui_ImplVulkan_DestroyFontUploadObjects();
		}
	}
	Module::~Module()
	{
		vkDeviceWaitIdle(Vulkan::Instance::get().logicalDevice());

		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}

	int Module::run(std::function<void()> update, std::function<void()> render)
	{
		while (!glfwWindowShouldClose(m_glfw.window()))
		{
			// Poll and handle events (inputs, window resize, etc.)
			// You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use 
			// your inputs.
			// - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main Rendering, 
			// or clear/overwrite your copy of the mouse data.
			// - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main Rendering, 
			// or clear/overwrite your copy of the keyboard data.
			// Generally you may always pass all inputs to dear imgui, 
			// and hide them from your Rendering based on those two flags.
			glfwPollEvents();

			// Resize swap chain
			if (m_vulkan.m_window.rebuild())
			{
				auto window_size = m_glfw.frameBufferSize();
				if (window_size.x > 0 && window_size.y > 0)
				{
					m_vulkan.window().reset(window_size);
					ImGui_ImplVulkan_SetMinImageCount(m_min_image_count);
				}
			}

			update();

			if (!m_glfw.iconified())
			{
				m_vulkan.m_window.acquireNextImage();
				m_vulkan.m_window.resetCommandPool();
				vkDeviceWaitIdle(Vulkan::Instance::get().logicalDevice());

				// Start the Dear ImGui frame
				ImGui_ImplVulkan_NewFrame();
				ImGui_ImplGlfw_NewFrame();
				ImGui::NewFrame();

				ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

				render();

				static bool show_demo = false;
				if (ImGui::IsKeyPressed(ImGuiKey_P) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl)) show_demo = true;
				if (show_demo) ImGui::ShowDemoWindow(&show_demo);


				{
					ImGui::Begin("Hello, world!");
					ImGui::Text(
						"Rendering average %.3f ms/frame (%.1f FPS)",
						1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
					ImGui::End();
				}

				ImGui::Render();

				m_vulkan.frameRender();

				// Update and Render additional Platform Windows
				if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
				{
					ImGui::UpdatePlatformWindows();
					ImGui::RenderPlatformWindowsDefault();
				}

				// Present Main Platform Window
				m_vulkan.m_window.framePresent();
			}
		}

		vkDeviceWaitIdle(Vulkan::Instance::get().logicalDevice());

		return 0;
	}
	void Module::setWindowTitle(const std::string& title)
	{
		m_glfw.setTitle(title);
	}
	void Module::setImguiStyle()
	{
		ImGuiIO& io = ImGui::GetIO();
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;           // Enable Docking
		io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows
		io.ConfigWindowsMoveFromTitleBarOnly = true;

		ImGui::StyleColorsDark();

		ImGui::GetStyle().ScrollbarRounding = 0.0f;
		ImGui::GetStyle().TabRounding = 0.0f;
		ImGui::GetStyle().WindowRounding = 0.0f;
		ImGui::GetStyle().Colors[ImGuiCol_WindowBg].w = 1.0f;

		ImVec4* colors = ImGui::GetStyle().Colors;
		colors[ImGuiCol_Text] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
		colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
		colors[ImGuiCol_WindowBg] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
		colors[ImGuiCol_ChildBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
		colors[ImGuiCol_PopupBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.94f);
		colors[ImGuiCol_Border] = ImVec4(0.50f, 0.50f, 0.50f, 0.50f);
		colors[ImGuiCol_BorderShadow] = ImVec4(0.25f, 0.25f, 0.25f, 0.00f);
		colors[ImGuiCol_FrameBg] = ImVec4(0.05f, 0.20f, 0.05f, 1.00f);
		colors[ImGuiCol_FrameBgHovered] = ImVec4(0.19f, 0.38f, 0.19f, 1.00f);
		colors[ImGuiCol_FrameBgActive] = ImVec4(0.25f, 0.50f, 0.25f, 1.00f);
		colors[ImGuiCol_TitleBg] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
		colors[ImGuiCol_TitleBgActive] = ImVec4(0.12f, 0.25f, 0.12f, 1.00f);
		colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.12f, 0.25f, 0.12f, 0.50f);
		colors[ImGuiCol_MenuBarBg] = ImVec4(0.06f, 0.25f, 0.06f, 1.00f);
		colors[ImGuiCol_ScrollbarBg] = ImVec4(0.02f, 0.02f, 0.02f, 0.50f);
		colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
		colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
		colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
		colors[ImGuiCol_CheckMark] = ImVec4(0.38f, 0.77f, 0.38f, 1.00f);
		colors[ImGuiCol_SliderGrab] = ImVec4(0.28f, 0.56f, 0.28f, 1.00f);
		colors[ImGuiCol_SliderGrabActive] = ImVec4(0.38f, 0.77f, 0.38f, 1.00f);
		colors[ImGuiCol_Button] = ImVec4(0.12f, 0.25f, 0.12f, 1.00f);
		colors[ImGuiCol_ButtonHovered] = ImVec4(0.19f, 0.38f, 0.19f, 1.00f);
		colors[ImGuiCol_ButtonActive] = ImVec4(0.25f, 0.50f, 0.25f, 1.00f);
		colors[ImGuiCol_Header] = ImVec4(0.09f, 0.38f, 0.18f, 1.00f);
		colors[ImGuiCol_HeaderHovered] = ImVec4(0.25f, 0.50f, 0.25f, 1.00f);
		colors[ImGuiCol_HeaderActive] = ImVec4(0.31f, 0.63f, 0.33f, 1.00f);
		colors[ImGuiCol_Separator] = ImVec4(0.50f, 0.50f, 0.25f, 0.50f);
		colors[ImGuiCol_SeparatorHovered] = ImVec4(0.19f, 0.38f, 0.19f, 0.78f);
		colors[ImGuiCol_SeparatorActive] = ImVec4(0.25f, 0.50f, 0.25f, 1.00f);
		colors[ImGuiCol_ResizeGrip] = ImVec4(0.12f, 0.25f, 0.12f, 0.50f);
		colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.19f, 0.38f, 0.19f, 0.50f);
		colors[ImGuiCol_ResizeGripActive] = ImVec4(0.25f, 0.50f, 0.25f, 0.50f);
		colors[ImGuiCol_Tab] = ImVec4(0.13f, 0.25f, 0.13f, 1.00f);
		colors[ImGuiCol_TabHovered] = ImVec4(0.25f, 0.50f, 0.25f, 1.00f);
		colors[ImGuiCol_TabActive] = ImVec4(0.19f, 0.38f, 0.19f, 1.00f);
		colors[ImGuiCol_TabUnfocused] = ImVec4(0.00f, 0.13f, 0.00f, 0.97f);
		colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.00f, 0.25f, 0.00f, 1.00f);
		colors[ImGuiCol_DockingPreview] = ImVec4(0.00f, 0.50f, 0.50f, 0.50f);
		colors[ImGuiCol_DockingEmptyBg] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
		colors[ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
		colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
		colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
		colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
		colors[ImGuiCol_TableHeaderBg] = ImVec4(0.19f, 0.20f, 0.19f, 1.00f);
		colors[ImGuiCol_TableBorderStrong] = ImVec4(0.31f, 0.35f, 0.31f, 1.00f);
		colors[ImGuiCol_TableBorderLight] = ImVec4(0.23f, 0.25f, 0.23f, 1.00f);
		colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
		colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.00f, 1.00f, 1.00f, 0.06f);
		colors[ImGuiCol_TextSelectedBg] = ImVec4(0.00f, 0.50f, 0.13f, 0.50f);
		colors[ImGuiCol_DragDropTarget] = ImVec4(1.00f, 1.00f, 0.00f, 0.77f);
		colors[ImGuiCol_NavHighlight] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
		colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
		colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
		colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
	}
}