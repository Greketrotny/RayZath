#include "cuda_render_parts.cuh"
#include "rzexception.h"

namespace RayZath
{
	//// ~~~~~~~~ [STRUCT] CudaMaterial ~~~~~~~~
	//CudaMaterial::CudaMaterial()
	//	: type(MaterialType::Diffuse)
	//	, glossiness(0.0f)
	//	, emission(0.0f)
	//	, reflectance(1.0f)
	//{}

	//CudaMaterial& CudaMaterial::operator=(const Material& material)
	//{
	//	this->type = material.GetMaterialType();
	//	this->glossiness = material.GetGlossiness();
	//	this->emission = material.GetEmitance();
	//	this->reflectance = material.GetReflectance();

	//	return *this;
	//}
	//// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 


	//
	//// ~~~~~~~~ [STRUCT] CudaTexture ~~~~~~~~
	//cudaChannelFormatDesc CudaTexture::chanelDesc = cudaCreateChannelDesc<uchar4>();

	//CudaTexture::CudaTexture()
	//	: textureArray(nullptr)
	//	, textureObject(0)
	//{}
	//CudaTexture::~CudaTexture()
	//{
	//	if (textureObject) CudaErrorCheck(cudaDestroyTextureObject(textureObject));
	//	if (textureArray)  CudaErrorCheck(cudaFreeArray(textureArray));

	//	this->textureObject = 0;
	//	this->textureArray = nullptr;
	//}

	//void CudaTexture::Reconstruct(const Texture& host_texture, cudaStream_t* mirror_stream)
	//{
	//	if (this->textureArray == nullptr)
	//	{//--> hostMesh has texture but device equivalent doesn't

	//		// texture array allocation
	//		CudaErrorCheck(cudaMallocArray(
	//			&this->textureArray,
	//			&this->chanelDesc,
	//			host_texture.Bitmap.GetWidth(), host_texture.Bitmap.GetHeight()));

	//		// copy host texture data to device array
	//		CudaErrorCheck(cudaMemcpyToArray(
	//			this->textureArray,
	//			0, 0, host_texture.Bitmap.GetMapAddress(), 
	//			host_texture.Bitmap.GetWidth() * host_texture.Bitmap.GetHeight() * sizeof(Graphics::Color),
	//			cudaMemcpyKind::cudaMemcpyHostToDevice));

	//		// specify resource description			
	//		memset(&this->resDesc, 0, sizeof(cudaResourceDesc));
	//		this->resDesc.resType = cudaResourceType::cudaResourceTypeArray;
	//		this->resDesc.res.array.array = this->textureArray;

	//		// specify texture object parameters
	//		memset(&this->textureDesc, 0, (sizeof(cudaTextureDesc)));
	//		this->textureDesc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeWrap;
	//		this->textureDesc.addressMode[1] = cudaTextureAddressMode::cudaAddressModeWrap;
	//		if (host_texture.FilterMode == Texture::TextureFilterModePoint)	this->textureDesc.filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	//		else															this->textureDesc.filterMode = cudaTextureFilterMode::cudaFilterModeLinear;
	//		this->textureDesc.readMode = cudaTextureReadMode::cudaReadModeNormalizedFloat;
	//		this->textureDesc.normalizedCoords = 1;

	//		// craete texture object
	//		CudaErrorCheck(cudaCreateTextureObject(
	//			&this->textureObject,
	//			&this->resDesc,
	//			&this->textureDesc,
	//			nullptr));
	//	}
	//	else
	//	{//--> Both hostMesh and deviceMesh have texture

	//		// get texture array info (width and height)
	//		cudaExtent arrayInfo;
	//		CudaErrorCheck(cudaArrayGetInfo(nullptr, &arrayInfo, nullptr, this->textureArray));

	//		if (arrayInfo.width * arrayInfo.height != host_texture.Bitmap.GetWidth() * host_texture.Bitmap.GetHeight())
	//		{//--> size of hostMesh texture and CudaMesh texture doesn't match

	//			// free CudaMesh array
	//			CudaErrorCheck(cudaFreeArray(this->textureArray));

	//			// array allocation
	//			CudaErrorCheck(cudaMallocArray(
	//				&this->textureArray,
	//				&this->chanelDesc,
	//				host_texture.Bitmap.GetWidth(), host_texture.Bitmap.GetHeight()));
	//			this->resDesc.res.array.array = this->textureArray;

	//			// copy host texture data to device array
	//			CudaErrorCheck(cudaMemcpyToArray(
	//				this->textureArray,
	//				0, 0, host_texture.Bitmap.GetMapAddress(), 
	//				host_texture.Bitmap.GetWidth() * host_texture.Bitmap.GetHeight()* sizeof(Graphics::Color),
	//				cudaMemcpyKind::cudaMemcpyHostToDevice));
	//		}
	//		else
	//		{//--> Everything does match so do asynchronous texture update (TODO)

	//			// copy host texture data to device array
	//			CudaErrorCheck(cudaMemcpyToArray(
	//				this->textureArray,
	//				0, 0, host_texture.Bitmap.GetMapAddress(), 
	//				host_texture.Bitmap.GetWidth() * host_texture.Bitmap.GetHeight() * sizeof(Graphics::Color),
	//				cudaMemcpyKind::cudaMemcpyHostToDevice));
	//		}
	//	}
	//}
	//// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}