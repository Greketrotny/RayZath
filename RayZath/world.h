#ifndef WORLD_H
#define WORLD_H

#include "world_object.h"

#include "camera.h"

#include "point_light.h"
#include "spot_light.h"
#include "direct_light.h"

#include "mesh.h"
#include "sphere.h"

namespace RayZath
{
	class World : public Updatable
	{
	public:
		template <class T> struct ObjectContainer : public Updatable
		{
		private:
			size_t m_count, m_capacity;
			T* mp_storage = nullptr;
			T** mpp_storage_ptr = nullptr;


		private:
			ObjectContainer(Updatable* updatable, size_t capacity = 16u)
				: Updatable(updatable)
			{
				this->m_count = 0u;
				this->m_capacity = std::max(capacity, size_t(2));

				mp_storage = (T*)malloc(this->m_capacity * sizeof(T));
				mpp_storage_ptr = (T**)malloc(this->m_capacity * sizeof(T*));
				for (size_t i = 0u; i < this->m_capacity; ++i)
					mpp_storage_ptr[i] = nullptr;
			}
			~ObjectContainer()
			{
				for (size_t i = 0u; i < m_capacity; ++i)
				{
					if (mpp_storage_ptr[i])
					{
						mp_storage[i].~T();
						mpp_storage_ptr[i] = nullptr;
					}
				}

				if (mp_storage) free(mp_storage);
				mp_storage = nullptr;
				if (mpp_storage_ptr) free(mpp_storage_ptr);
				mpp_storage_ptr = nullptr;

				m_count = 0u;
				m_capacity = 0u;
			}


		public:
			T* operator[](size_t index)
			{
				return mpp_storage_ptr[index];
			}
			const T* operator[](size_t index) const
			{
				return mpp_storage_ptr[index];
			}


		public:
			T* CreateObject(const ConStruct<T>& conStruct = ConStruct<T>())
			{
				if (m_count >= m_capacity)
					return nullptr;

				T* newObject = nullptr;
				for (size_t i = 0u; i < m_capacity; ++i)
				{
					if (!mpp_storage_ptr[i])
					{
						new (&mp_storage[i]) T(i, this, conStruct);
						mpp_storage_ptr[i] = &mp_storage[i];
						newObject = &mp_storage[i];
						++m_count;
						return newObject;
					}
				}
				return newObject;
			}
			bool DestroyObject(const T* object)
			{
				if (m_count == 0u || object == nullptr)
					return false;

				for (size_t i = 0u; i < m_capacity; ++i)
				{
					if (mpp_storage_ptr[i] && object == &mp_storage[i])
					{
						mp_storage[i].~T();
						mpp_storage_ptr[i] = nullptr;
						--m_count;
						return true;
					}
				}
				return false;
			}
			void DestroyAllObjects()
			{
				for (size_t i = 0u; i < m_capacity; ++i)
				{
					if (mpp_storage_ptr[i])
					{
						mp_storage[i].~T();
						mpp_storage_ptr[i] = nullptr;
					}
				}
				m_count = 0u;
			}

			size_t GetCount() const
			{
				return m_count;
			}
			size_t GetCapacity() const
			{
				return	m_capacity;
			}


			friend class World;
		};
	private:
		ObjectContainer<Camera> m_cameras;

		ObjectContainer<PointLight> m_point_lights;
		ObjectContainer<SpotLight> m_spot_lights;
		ObjectContainer<DirectLight> m_direct_lights;

		ObjectContainer<Mesh> m_meshes;
		ObjectContainer<Sphere> m_spheres;


	private:
		World(
			size_t maxCamerasCount = 16u, 
			size_t maxLightsCount = 16u, 
			size_t maxRenderObjectsCount = 1024u);
		~World();


	public:
		ObjectContainer<Camera>& GetCameras();
		const ObjectContainer<Camera>& GetCameras() const;

		ObjectContainer<PointLight>& GetPointLights();
		const ObjectContainer<PointLight>& GetPointLights() const;
		ObjectContainer<SpotLight>& GetSpotLights();
		const ObjectContainer<SpotLight>& GetSpotLights() const;
		ObjectContainer<DirectLight>& GetDirectLights();
		const ObjectContainer<DirectLight>& GetDirectLights() const;

		ObjectContainer<Mesh>& GetMeshes();
		const ObjectContainer<Mesh>& GetMeshes() const;
		ObjectContainer<Sphere>& GetSpheres();
		const ObjectContainer<Sphere>& GetSpheres() const;


		void DestroyAllComponents();


		friend class Engine;
	};
}

#endif // !WORLD_H