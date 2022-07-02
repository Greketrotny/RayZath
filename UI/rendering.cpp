#include "rendering.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include "glfw.hpp"
#include "vulkan.hpp"

#include "rayzath.h"

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

	RenderingWrapper::RenderingWrapper()
		: m_vulkan(m_glfw)
	{
		m_glfw.init();
		m_vulkan.init();

		// Setup Dear ImGui context
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;           // Enable Docking
		io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows
		io.ConfigWindowsMoveFromTitleBarOnly = true;

		// Setup Dear ImGui style
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
		colors[ImGuiCol_FrameBg] = ImVec4(0.12f, 0.25f, 0.12f, 1.00f);
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
		colors[ImGuiCol_Header] = ImVec4(0.12f, 0.25f, 0.16f, 1.00f);
		colors[ImGuiCol_HeaderHovered] = ImVec4(0.19f, 0.38f, 0.19f, 1.00f);
		colors[ImGuiCol_HeaderActive] = ImVec4(0.25f, 0.50f, 0.27f, 1.00f);
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

		// Setup Platform/Renderer backends
		ImGui_ImplGlfw_InitForVulkan(m_glfw.window(), true);
		ImGui_ImplVulkan_InitInfo init_info = {};
		init_info.Instance = m_vulkan.instance().vulkanInstance();
		init_info.PhysicalDevice = m_vulkan.instance().physicalDevice();
		init_info.Device = m_vulkan.instance().logicalDevice();
		init_info.QueueFamily = m_vulkan.instance().queueFamilyIdx();
		init_info.Queue = m_vulkan.instance().queue();
		init_info.PipelineCache = VK_NULL_HANDLE;
		init_info.DescriptorPool = m_vulkan.instance().descriptorPool();
		init_info.Subpass = 0;
		init_info.MinImageCount = m_vulkan.window().m_min_image_count;
		init_info.ImageCount = init_info.MinImageCount;
		init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
		init_info.Allocator = m_vulkan.instance().allocator();
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
			VkCommandBuffer command_buffer = m_vulkan.m_window.currentFrame().m_command_buffer;

			Vulkan::check(vkResetCommandPool(m_vulkan.instance().logicalDevice(), command_pool, 0));
			VkCommandBufferBeginInfo begin_info{};
			begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			Vulkan::check(vkBeginCommandBuffer(command_buffer, &begin_info));

			ImGui_ImplVulkan_CreateFontsTexture(command_buffer);

			Graphics::Bitmap bitmap(123, 123);
			for (size_t y = 0; y < bitmap.GetWidth(); y++)
			{
				for (size_t x = 0; x < bitmap.GetHeight(); x++)
				{
					bitmap.Value(x, y) = Graphics::Color(0x40, 0xFF, 0x40);
				}
			}
			m_vulkan.m_render_image.createImage(bitmap, command_buffer);


			VkSubmitInfo end_info = {};
			end_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			end_info.commandBufferCount = 1;
			end_info.pCommandBuffers = &command_buffer;
			Vulkan::check(vkEndCommandBuffer(command_buffer));
			Vulkan::check(vkQueueSubmit(m_vulkan.instance().queue(), 1, &end_info, VK_NULL_HANDLE));

			Vulkan::check(vkDeviceWaitIdle(m_vulkan.instance().logicalDevice()));
			ImGui_ImplVulkan_DestroyFontUploadObjects();
		}
	}
	RenderingWrapper::~RenderingWrapper()
	{
		vkDeviceWaitIdle(m_vulkan.instance().logicalDevice());

		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}

	int RenderingWrapper::run(std::function<void()> drawUi)
	{
		// Our state
		bool show_demo_window = true;
		bool show_another_window = false;
		ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

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

			m_vulkan.m_window.resetCommandPool();
			vkDeviceWaitIdle(m_vulkan.instance().logicalDevice());

			// Start the Dear ImGui frame
			ImGui_ImplVulkan_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

			drawUi();
			
			// 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
			//if (show_demo_window)
			//	ImGui::ShowDemoWindow(&show_demo_window);

			// 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
			{
				ImGui::Begin("Hello, world!");     
				ImGui::Text(
					"Rendering average %.3f ms/frame (%.1f FPS)", 
					1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
				ImGui::End();
			}

			// Rendering
			ImGui::Render();
			ImDrawData* main_draw_data = ImGui::GetDrawData();
			const bool main_is_minimized = (main_draw_data->DisplaySize.x <= 0.0f || main_draw_data->DisplaySize.y <= 0.0f);
			if (!main_is_minimized)
				m_vulkan.frameRender();

			// Update and Render additional Platform Windows
			if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
			{
				ImGui::UpdatePlatformWindows();
				ImGui::RenderPlatformWindowsDefault();
			}

			// Present Main Platform Window
			if (!main_is_minimized)
				m_vulkan.m_window.framePresent();
		}

		vkDeviceWaitIdle(m_vulkan.instance().logicalDevice());

		return 0;
	}
}