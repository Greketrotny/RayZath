#ifndef RENDER_PARTS_H
#define RENDER_PARTS_H

#include "cuda_include.h"

//#include "vec3.h"
//#include "color.h"
//#include "bitmap.h"

namespace RayZath
{
	enum class MaterialType
	{
		Diffuse,
		Glossy,
		Specular,
		Refractive,
		Transparent,
		Light
	};

	/*struct Material
	{
	private:
		MaterialType m_material_type;
		float m_glossiness;
		float m_emitance;
		float m_reflectance;


	public:
		Material(
			MaterialType type = MaterialType::Diffuse,
			float glossiness = 0.0f,
			float emitance = 0.0f,
			float reflectance = 1.0f);
		Material(const Material& material);
		~Material();


	public:
		Material& operator=(const Material& material);


	public:
		void Set(MaterialType type, float glossiness, float emitance, float reflectance);
		void SetMaterialType(MaterialType type);
		void SetGlossiness(const float glossiness);
		void SetEmitance(const float emitance);
		void SetReflectance(const float reflectance);

		MaterialType GetMaterialType() const;
		float GetGlossiness() const;
		float GetEmitance() const;
		float GetReflectance() const;
	};*/

	/*struct Texcrd
	{
	public:
		float u, v;


	public:
		Texcrd(float u, float v)
			: u(u)
			, v(v)
		{}
	};
	struct Texture
	{
	public:
		Graphics::Bitmap Bitmap;
		enum TextureFilterMode
		{
			TextureFilterModePoint,
			TextureFilterModeLinear
		};
		TextureFilterMode FilterMode = TextureFilterMode::TextureFilterModePoint;


	public:
		Texture() = delete;
		Texture(const Texture& texture);
		Texture(Texture&& texture);
		Texture(unsigned int width, unsigned int height, TextureFilterMode filterMode = TextureFilterModePoint);
		Texture(const Graphics::Bitmap& bitmap, TextureFilterMode filterMode = TextureFilterModePoint);
		~Texture();


	public:
		Texture& operator=(const Texture& texture);
		Texture& operator=(Texture&& texture);


	public:
		friend struct CudaTexture;
	};*/

	//struct Triangle
	//{
	//	typedef Math::vec3<float> Vertex;
	//private:
	//	Vertex* v1 = nullptr, * v2 = nullptr, * v3 = nullptr;
	//	Vertex normal;
	//	Texcrd* t1 = nullptr, * t2 = nullptr, * t3 = nullptr;
	//	Graphics::Color color;


	//private:
	//	Triangle(
	//		Vertex* v1, Vertex* v2, Vertex* v3,
	//		Texcrd* t1, Texcrd* t2, Texcrd* t3,
	//		Graphics::Color color = Graphics::Color(0xFF, 0xFF, 0xFF));
	//	~Triangle();


	//public:
	//	// overall
	//	const Graphics::Color& Color();
	//	void Color(const Graphics::Color& newColor);
	//	const Math::vec3<float>& GetNormal() const;

	//	// vertices
	//	Vertex& V1() const;
	//	void V1(const Vertex& newVertex);
	//	void V1(const float& x, const float& y, const float& z);
	//	Vertex& V2() const;
	//	void V2(const Vertex& newVertex);
	//	void V2(const float& x, const float& y, const float& z);
	//	Vertex& V3() const;
	//	void V3(const Vertex& newVertex);
	//	void V3(const float& x, const float& y, const float& z);

	//	// texture coordinates
	//	Texcrd& T1() const;
	//	void T1(const Texcrd& newTexcds);
	//	void T1(const float& u, const float& v);
	//	Texcrd& T2() const;
	//	void T2(const Texcrd& newTexcds);
	//	void T2(const float& u, const float& v);
	//	Texcrd& T3() const;
	//	void T3(const Texcrd& newTexcds);
	//	void T3(const float& u, const float& v);

	//	// -- friends -- //
	//	friend class Mesh;
	//	friend class CudaMesh;
	//	friend struct CudaTriangleStorage;
	//	friend struct CudaTriangle;
	//};
}

#endif // !RENDER_PARTS_H