module;

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include "glfw3.h"
#include "vulkan/vulkan.h"

#include "rayzath.h"

#include <stdexcept>
#include <iostream>
#include <string>
#include <functional>

module rz.ui.rendering;

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

		// Setup Dear ImGui style
		ImGui::StyleColorsDark();

		// When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
		ImGuiStyle& style = ImGui::GetStyle();
		if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			style.WindowRounding = 0.0f;
			style.Colors[ImGuiCol_WindowBg].w = 1.0f;

		}

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
			VkCommandBufferBeginInfo begin_info = {};
			begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			Vulkan::check(vkBeginCommandBuffer(command_buffer, &begin_info));

			ImGui_ImplVulkan_CreateFontsTexture(command_buffer);

			Graphics::Bitmap bitmap(200, 200);
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

			// Resize swap chain?
			if (m_vulkan.m_window.rebuild())
			{
				auto window_size = m_glfw.frameBufferSize();
				if (window_size.x > 0 && window_size.y > 0)
				{
					m_vulkan.window().reset(window_size);
					ImGui_ImplVulkan_SetMinImageCount(m_min_image_count);
				}
			}

			// Start the Dear ImGui frame
			ImGui_ImplVulkan_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

			//Graphics::Bitmap bitmap(200, 200);
			//for (size_t y = 0; y < bitmap.GetWidth(); y++)
			//{
			//	for (size_t x = 0; x < bitmap.GetHeight(); x++)
			//	{
			//		bitmap.Value(x, y) = Graphics::Color(clear_color.x, clear_color.y, clear_color.z);
			//	}
			//}

			//Render::Vulkan::check(vkDeviceWaitIdle(m_vulkan.m_logical_device));
			//// Use any command queue
			//VkCommandPool command_pool =
			//	m_vulkan.m_imgui_main_window.Frames[m_vulkan.m_imgui_main_window.FrameIndex].CommandPool;
			//VkCommandBuffer command_buffer =
			//	m_vulkan.m_imgui_main_window.Frames[m_vulkan.m_imgui_main_window.FrameIndex].CommandBuffer;

			//Render::Vulkan::check(vkResetCommandPool(m_vulkan.m_logical_device, command_pool, 0));
			//VkCommandBufferBeginInfo begin_info = {};
			//begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			//begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			//Render::Vulkan::check(vkBeginCommandBuffer(command_buffer, &begin_info));

			//m_vulkan.updateImage(bitmap, command_buffer);

			//VkSubmitInfo end_info = {};
			//end_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			//end_info.commandBufferCount = 1;
			//end_info.pCommandBuffers = &command_buffer;
			//Render::Vulkan::check(vkEndCommandBuffer(command_buffer));
			//Render::Vulkan::check(vkQueueSubmit(m_vulkan.m_queue, 1, &end_info, VK_NULL_HANDLE));

			drawUi();

			
			// 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
			//if (show_demo_window)
			//	ImGui::ShowDemoWindow(&show_demo_window);

			// 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
			{
				static float f = 0.0f;
				static int counter = 0;

				ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

				ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
				ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
				ImGui::Checkbox("Another Window", &show_another_window);

				ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
				ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

				if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
					counter++;
				ImGui::SameLine();
				ImGui::Text("counter = %d", counter);

				ImGui::Text("Rendering average %.3f ms/frame (%.1f FPS)", 
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