#ifndef RZ_TYPEDEFS_HPP
#define RZ_TYPEDEFS_HPP

#include "dictionary.hpp"

namespace RayZath::Engine
{
	enum class ObjectType
	{
		Texture,
		NormalMap,
		MetalnessMap,
		RoughnessMap,
		EmissionMap,

		Material,
		Mesh,

		Camera,

		SpotLight,
		DirectLight,

		Instance,

		Group,
	};

	template <ObjectType T>
	static constexpr std::size_t idx_of = Utils::static_dictionary::vv_translate<T>::template with<
		Utils::static_dictionary::vv_translation<ObjectType::Texture, 0>,
		Utils::static_dictionary::vv_translation<ObjectType::NormalMap, 1>,
		Utils::static_dictionary::vv_translation<ObjectType::MetalnessMap, 2>,
		Utils::static_dictionary::vv_translation<ObjectType::RoughnessMap, 3>,
		Utils::static_dictionary::vv_translation<ObjectType::EmissionMap, 4>,
		Utils::static_dictionary::vv_translation<ObjectType::Material, 5>,
		Utils::static_dictionary::vv_translation<ObjectType::Mesh, 6>,
		Utils::static_dictionary::vv_translation<ObjectType::Camera, 7>,
		Utils::static_dictionary::vv_translation<ObjectType::SpotLight, 8>,
		Utils::static_dictionary::vv_translation<ObjectType::DirectLight, 9>,
		Utils::static_dictionary::vv_translation<ObjectType::Instance, 10>,
		Utils::static_dictionary::vv_translation<ObjectType::Group, 11>>::value;
}

#endif 
