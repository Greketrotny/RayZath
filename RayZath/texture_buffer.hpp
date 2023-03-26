#ifndef TEXTURE_BUFFER_HPP
#define TEXTURE_BUFFER_HPP

#include "world_object.hpp"
#include "bitmap.h"	
#include "vec2.h"
#include "angle.h"

namespace RayZath::Engine
{
	struct TextureBufferBase
	{
		enum class FilterMode
		{
			Point,
			Linear
		};
		enum class AddressMode
		{
			Wrap,
			Clamp,
			Mirror,
			Border
		};
	};

	template <typename T>
	struct TextureBuffer
		: public WorldObject
		, public TextureBufferBase
	{
	public:
		using buffer_t = Graphics::Buffer2D<T>;
	private:
		buffer_t m_bitmap;
		FilterMode m_filter_mode;
		AddressMode m_address_mode;
		Math::vec2f m_scale;
		Math::angle_radf m_rotation;
		Math::vec2f m_translation;


	public:
		TextureBuffer(const TextureBuffer& bitmap) = delete;
		TextureBuffer(TextureBuffer&& bitmap) = delete;
		TextureBuffer(
			Updatable* updatable,
			ConStruct<TextureBuffer<T>> con_struct)
			: WorldObject(updatable, con_struct)
			, m_bitmap(std::move(con_struct.bitmap))
			, m_filter_mode(std::move(con_struct.filter_mode))
			, m_address_mode(std::move(con_struct.address_mode))
			, m_scale(std::move(con_struct.scale))
			, m_rotation(std::move(con_struct.rotation))
			, m_translation(std::move(con_struct.translation))
		{}


	public:
		TextureBuffer& operator=(const TextureBuffer& bitmap) = delete;
		TextureBuffer& operator=(TextureBuffer&& bitmap) = delete;


	public:
		auto& bitmap() noexcept
		{
			return m_bitmap;
		}
		const auto& bitmap() const noexcept
		{
			return m_bitmap;
		}
		FilterMode filterMode() const noexcept
		{
			return m_filter_mode;
		}
		AddressMode addressMode() const noexcept
		{
			return m_address_mode;
		}
		Math::vec2f scale() const
		{
			return m_scale;
		}
		Math::angle_radf rotation() const
		{
			return m_rotation;
		}
		Math::vec2f translation() const
		{
			return m_translation;
		}

		void bitmap(const buffer_t& bitmap)
		{
			m_bitmap = bitmap;
			stateRegister().RequestUpdate();
		}
		void filterMode(const FilterMode filter_mode)
		{
			m_filter_mode = filter_mode;
			stateRegister().RequestUpdate();
		}
		void addressMode(const AddressMode address_mode)
		{
			m_address_mode = address_mode;
			stateRegister().RequestUpdate();
		}
		void scale(const Math::vec2f& scale)
		{
			m_scale = scale;
			stateRegister().RequestUpdate();
		}
		void rotation(const Math::angle_radf& rotation)
		{
			m_rotation = rotation;
			stateRegister().RequestUpdate();
		}
		void translation(const Math::vec2f& translation)
		{
			m_translation = translation;
			stateRegister().RequestUpdate();
		}

		const auto& fetch(Math::vec2f32 texcrd) const
		{
			texcrd += m_translation;
			texcrd.Rotate(m_rotation.value());
			texcrd *= m_scale;

			texcrd.x = std::fmodf(std::fmodf(texcrd.x, 1.0f) + 1.0f, 1.0f);
			texcrd.y = 1.0f - std::fmodf(std::fmodf(texcrd.y, 1.0f) + 1.0f, 1.0f);

			return bitmap().Value(
				std::clamp(size_t(texcrd.x * bitmap().GetWidth()), size_t(0), size_t(bitmap().GetWidth() - 1u)),
				std::clamp(size_t(texcrd.y * bitmap().GetHeight()), size_t(0), size_t(bitmap().GetHeight() - 1u)));
		}
	};
	using Texture = TextureBuffer<Graphics::Color>;
	using NormalMap = TextureBuffer<Graphics::Color>;
	using MetalnessMap = TextureBuffer<uint8_t>;
	using RoughnessMap = TextureBuffer<uint8_t>;
	using EmissionMap = TextureBuffer<float>;

	template <typename T>
	struct ConStruct<TextureBuffer<T>>
		: public ConStruct<WorldObject>
	{
		Graphics::Buffer2D<T> bitmap;
		typename TextureBuffer<T>::FilterMode filter_mode;
		typename TextureBuffer<T>::AddressMode address_mode;
		Math::vec2f scale;
		Math::angle_radf rotation;
		Math::vec2f translation;

		ConStruct(
			std::string name = "name",
			Graphics::Buffer2D<T> bitmap = Graphics::Buffer2D<T>(16u, 16u),
			typename TextureBuffer<T>::FilterMode filter_mode = TextureBuffer<T>::FilterMode::Point,
			typename TextureBuffer<T>::AddressMode address_mode = TextureBuffer<T>::AddressMode::Wrap,
			Math::vec2f scale = Math::vec2f(1.0f, 1.0f),
			Math::angle_radf rotation = Math::angle_radf(0.0f),
			Math::vec2f translation = Math::vec2f(0.0f, 0.0f))
			: ConStruct<WorldObject>(name)
			, bitmap(std::move(bitmap))
			, filter_mode(std::move(filter_mode))
			, address_mode(std::move(address_mode))
			, scale(std::move(scale))
			, rotation(std::move(rotation))
			, translation(std::move(translation))
		{}
	};
}

#endif
