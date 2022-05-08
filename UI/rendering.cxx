module;

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include "glfw3.h"
#include "vulkan/vulkan.h"

#include "rayzath.h"


module rz.ui.rendering;

import rz.ui.application;

namespace RayZath::UI
{
	static void check_vk_result(VkResult err)
	{
		if (err != 0)
			throw std::exception((std::string("[vulkan] Error: VkResult = ") + std::to_string(err)).c_str());
	}

	Rendering::Rendering()
		: m_glfw(m_vulkan)
		, m_vulkan(m_glfw)
	{
		// initiate vulkan
		uint32_t extensions_count = 0;
		const char** extensions = glfwGetRequiredInstanceExtensions(&extensions_count);
		m_vulkan.createInstance(extensions, extensions_count);
		m_vulkan.selectPhysicalDevice();
		m_vulkan.createLogicalDevice();
		m_vulkan.createDescriptorPool();

		// Create Window Surface
		VkSurfaceKHR surface;
		
		VkResult err = glfwCreateWindowSurface(m_vulkan.m_instance, m_glfw.window(), m_vulkan.mp_allocator, &surface);
		check_vk_result(err);

		// Create Framebuffers
		int w, h;
		glfwGetFramebufferSize(m_glfw.window(), &w, &h);
		SetupVulkanWindow(&m_imgui_main_window, surface, w, h);

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
		init_info.Instance = m_vulkan.m_instance;
		init_info.PhysicalDevice = m_vulkan.m_physical_device;
		init_info.Device = m_vulkan.m_logical_device;
		init_info.QueueFamily = m_vulkan.m_queue_family_idx;
		init_info.Queue = m_vulkan.m_queue;
		init_info.PipelineCache = m_vk_pipeline_cache;
		init_info.DescriptorPool = m_vulkan.m_descriptor_pool;
		init_info.Subpass = 0;
		init_info.MinImageCount = m_min_image_count;
		init_info.ImageCount = m_imgui_main_window.ImageCount;
		init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
		init_info.Allocator = m_vulkan.mp_allocator;
		init_info.CheckVkResultFn = check_vk_result;
		ImGui_ImplVulkan_Init(&init_info, m_imgui_main_window.RenderPass);

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
			VkCommandPool command_pool = m_imgui_main_window.Frames[m_imgui_main_window.FrameIndex].CommandPool;
			VkCommandBuffer command_buffer = m_imgui_main_window.Frames[m_imgui_main_window.FrameIndex].CommandBuffer;

			err = vkResetCommandPool(m_vulkan.m_logical_device, command_pool, 0);
			check_vk_result(err);
			VkCommandBufferBeginInfo begin_info = {};
			begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			err = vkBeginCommandBuffer(command_buffer, &begin_info);
			check_vk_result(err);

			ImGui_ImplVulkan_CreateFontsTexture(command_buffer);

			VkSubmitInfo end_info = {};
			end_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			end_info.commandBufferCount = 1;
			end_info.pCommandBuffers = &command_buffer;
			err = vkEndCommandBuffer(command_buffer);
			check_vk_result(err);
			err = vkQueueSubmit(m_vulkan.m_queue, 1, &end_info, VK_NULL_HANDLE);
			check_vk_result(err);

			err = vkDeviceWaitIdle(m_vulkan.m_logical_device);
			check_vk_result(err);
			ImGui_ImplVulkan_DestroyFontUploadObjects();
		}
	}
	Rendering::~Rendering()
	{
		vkDeviceWaitIdle(m_vulkan.m_logical_device);

		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();

		CleanupVulkanWindow();
		CleanupVulkan();

		glfwDestroyWindow(m_glfw.window());
		glfwTerminate();
	}

	int Rendering::run()
	{
		// Our state
		bool show_demo_window = true;
		bool show_another_window = false;
		ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

		while (!glfwWindowShouldClose(m_glfw.window()))
		{
			// Poll and handle events (inputs, window resize, etc.)
			// You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
			// - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main Rendering, or clear/overwrite your copy of the mouse data.
			// - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main Rendering, or clear/overwrite your copy of the keyboard data.
			// Generally you may always pass all inputs to dear imgui, and hide them from your Rendering based on those two flags.
			glfwPollEvents();

			// Resize swap chain?
			if (g_SwapChainRebuild)
			{
				int width, height;
				glfwGetFramebufferSize(m_glfw.window(), &width, &height);
				if (width > 0 && height > 0)
				{
					ImGui_ImplVulkan_SetMinImageCount(m_min_image_count);
					ImGui_ImplVulkanH_CreateOrResizeWindow(
						m_vulkan.m_instance,
						m_vulkan.m_physical_device,
						m_vulkan.m_logical_device,
						&m_imgui_main_window,
						m_vulkan.m_queue_family_idx,
						m_vulkan.mp_allocator,
						width, height,
						m_min_image_count);
					m_imgui_main_window.FrameIndex = 0;
					g_SwapChainRebuild = false;
				}
			}

			// Start the Dear ImGui frame
			ImGui_ImplVulkan_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

			Graphics::Bitmap bitmap(200, 200);
			for (size_t y = 0; y < bitmap.GetWidth(); y++)
			{
				for (size_t x = 0; x < bitmap.GetHeight(); x++)
				{
					bitmap.Value(x, y) = Graphics::Color(0xFF, 0x40, 0x40);
				}
			}

			auto& application = Application::instance();
			application.draw(std::move(bitmap));

			
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

				ImGui::Text("Rendering average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
				ImGui::End();
			}

			// Rendering
			ImGui::Render();
			ImDrawData* main_draw_data = ImGui::GetDrawData();
			const bool main_is_minimized = (main_draw_data->DisplaySize.x <= 0.0f || main_draw_data->DisplaySize.y <= 0.0f);
			m_imgui_main_window.ClearValue.color.float32[0] = clear_color.x * clear_color.w;
			m_imgui_main_window.ClearValue.color.float32[1] = clear_color.y * clear_color.w;
			m_imgui_main_window.ClearValue.color.float32[2] = clear_color.z * clear_color.w;
			m_imgui_main_window.ClearValue.color.float32[3] = clear_color.w;
			if (!main_is_minimized)
				FrameRender(&m_imgui_main_window, main_draw_data);

			// Update and Render additional Platform Windows
			if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
			{
				ImGui::UpdatePlatformWindows();
				ImGui::RenderPlatformWindowsDefault();
			}

			// Present Main Platform Window
			if (!main_is_minimized)
				FramePresent(&m_imgui_main_window);
		}

		return 0;
	}

	void Rendering::SetupVulkanWindow(ImGui_ImplVulkanH_Window* wd, VkSurfaceKHR surface, const int width, const int height)
	{
		m_imgui_main_window.Surface = surface;

		// Check for WSI support
		VkBool32 res;
		vkGetPhysicalDeviceSurfaceSupportKHR(m_vulkan.m_physical_device, m_vulkan.m_queue_family_idx, m_imgui_main_window.Surface, &res);
		if (res != VK_TRUE)
		{
			std::cerr << "Error no WSI support on physical device 0\n";
		}

		// Select Surface Format
		const VkFormat requestSurfaceImageFormat[] = {
			VK_FORMAT_B8G8R8A8_UNORM,
			VK_FORMAT_R8G8B8A8_UNORM,
			VK_FORMAT_B8G8R8_UNORM,
			VK_FORMAT_R8G8B8_UNORM };
		const VkColorSpaceKHR requestSurfaceColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
		m_imgui_main_window.SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(
			m_vulkan.m_physical_device,
			m_imgui_main_window.Surface,
			requestSurfaceImageFormat,
			(size_t)IM_ARRAYSIZE(requestSurfaceImageFormat),
			requestSurfaceColorSpace);

		// Select Present Mode
		#ifdef IMGUI_UNLIMITED_FRAME_RATE
		VkPresentModeKHR present_modes[] = { VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_FIFO_KHR };
		#else
		VkPresentModeKHR present_modes[] = { VK_PRESENT_MODE_FIFO_KHR };
		#endif
		m_imgui_main_window.PresentMode = ImGui_ImplVulkanH_SelectPresentMode(
			m_vulkan.m_physical_device,
			m_imgui_main_window.Surface,
			&present_modes[0],
			IM_ARRAYSIZE(present_modes));

		// Create SwapChain, RenderPass, Framebuffer, etc.
		IM_ASSERT(m_min_image_count >= 2);
		ImGui_ImplVulkanH_CreateOrResizeWindow(
			m_vulkan.m_instance,
			m_vulkan.m_physical_device,
			m_vulkan.m_logical_device,
			wd,
			m_vulkan.m_queue_family_idx,
			m_vulkan.mp_allocator,
			width, height,
			m_min_image_count);
	}
	void Rendering::CleanupVulkan()
	{
		m_vulkan.destroyDescriptorPool();
		m_vulkan.destroyLogicalDevice();
		m_vulkan.destroyInstance();
	}
	void Rendering::CleanupVulkanWindow()
	{
		ImGui_ImplVulkanH_DestroyWindow(m_vulkan.m_instance, m_vulkan.m_logical_device, &m_imgui_main_window, m_vulkan.mp_allocator);
	}

	void Rendering::FrameRender(ImGui_ImplVulkanH_Window* wd, ImDrawData* draw_data)
	{
		VkResult err;

		VkSemaphore image_acquired_semaphore = m_imgui_main_window.FrameSemaphores[m_imgui_main_window.SemaphoreIndex].ImageAcquiredSemaphore;
		VkSemaphore render_complete_semaphore = m_imgui_main_window.FrameSemaphores[m_imgui_main_window.SemaphoreIndex].RenderCompleteSemaphore;
		err = vkAcquireNextImageKHR(m_vulkan.m_logical_device, m_imgui_main_window.Swapchain, UINT64_MAX, image_acquired_semaphore, VK_NULL_HANDLE, &m_imgui_main_window.FrameIndex);
		if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
		{
			g_SwapChainRebuild = true;
			return;
		}
		check_vk_result(err);

		ImGui_ImplVulkanH_Frame* fd = &m_imgui_main_window.Frames[m_imgui_main_window.FrameIndex];
		{
			err = vkWaitForFences(m_vulkan.m_logical_device, 1, &fd->Fence, VK_TRUE, UINT64_MAX);    // wait indefinitely instead of periodically checking
			check_vk_result(err);

			err = vkResetFences(m_vulkan.m_logical_device, 1, &fd->Fence);
			check_vk_result(err);
		}
		{
			err = vkResetCommandPool(m_vulkan.m_logical_device, fd->CommandPool, 0);
			check_vk_result(err);
			VkCommandBufferBeginInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			err = vkBeginCommandBuffer(fd->CommandBuffer, &info);
			check_vk_result(err);
		}
		{
			VkRenderPassBeginInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			info.renderPass = m_imgui_main_window.RenderPass;
			info.framebuffer = fd->Framebuffer;
			info.renderArea.extent.width = m_imgui_main_window.Width;
			info.renderArea.extent.height = m_imgui_main_window.Height;
			info.clearValueCount = 1;
			info.pClearValues = &m_imgui_main_window.ClearValue;
			vkCmdBeginRenderPass(fd->CommandBuffer, &info, VK_SUBPASS_CONTENTS_INLINE);
		}

		// Record dear imgui primitives into command buffer
		ImGui_ImplVulkan_RenderDrawData(draw_data, fd->CommandBuffer);

		// Submit command buffer
		vkCmdEndRenderPass(fd->CommandBuffer);
		{
			VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			VkSubmitInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			info.waitSemaphoreCount = 1;
			info.pWaitSemaphores = &image_acquired_semaphore;
			info.pWaitDstStageMask = &wait_stage;
			info.commandBufferCount = 1;
			info.pCommandBuffers = &fd->CommandBuffer;
			info.signalSemaphoreCount = 1;
			info.pSignalSemaphores = &render_complete_semaphore;

			err = vkEndCommandBuffer(fd->CommandBuffer);
			check_vk_result(err);
			err = vkQueueSubmit(m_vulkan.m_queue, 1, &info, fd->Fence);
			check_vk_result(err);
		}
	}
	void Rendering::FramePresent(ImGui_ImplVulkanH_Window* wd)
	{
		if (g_SwapChainRebuild)
			return;
		VkSemaphore render_complete_semaphore = m_imgui_main_window.FrameSemaphores[m_imgui_main_window.SemaphoreIndex].RenderCompleteSemaphore;
		VkPresentInfoKHR info = {};
		info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		info.waitSemaphoreCount = 1;
		info.pWaitSemaphores = &render_complete_semaphore;
		info.swapchainCount = 1;
		info.pSwapchains = &m_imgui_main_window.Swapchain;
		info.pImageIndices = &m_imgui_main_window.FrameIndex;
		VkResult err = vkQueuePresentKHR(m_vulkan.m_queue, &info);
		if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
		{
			g_SwapChainRebuild = true;
			return;
		}
		check_vk_result(err);
		m_imgui_main_window.SemaphoreIndex = (m_imgui_main_window.SemaphoreIndex + 1) % m_imgui_main_window.ImageCount; // Now we can use the next set of semaphores
	}
}