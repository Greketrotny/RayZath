#include "cuda_render_parts.cuh"
#include "rzexception.h"

#include "curand.h"

namespace RayZath
{
	// ~~~~~~~~ [STRUCT] CudaMaterial ~~~~~~~~
	CudaMaterial& CudaMaterial::operator=(const Material& material)
	{
		this->reflectance = material.GetReflectance();
		this->glossiness = material.GetGlossiness();
		this->transmitance = material.GetTransmitance();
		this->ior = material.GetIndexOfRefraction();
		this->emitance = material.GetEmitance();
		return *this;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 



	// ~~~~~~~~ [STRUCT] RandomNumbers ~~~~~~~~
	HostPinnedMemory RandomNumbers::s_hpm(RandomNumbers::s_count * sizeof(*RandomNumbers::m_unsigned_uniform));

	RandomNumbers::RandomNumbers()
	{
		/*CudaErrorCheck(cudaMalloc(
			(void**)&m_unsigned_uniform, 
			RandomNumbers::s_count * sizeof(*m_unsigned_uniform)));

		CudaErrorCheck(cudaMalloc(
			(void**)&m_signed_uniform, 
			RandomNumbers::s_count * sizeof(*m_signed_uniform)));*/
	}
	RandomNumbers::~RandomNumbers()
	{
	/*	if (m_unsigned_uniform) CudaErrorCheck(cudaFree(m_unsigned_uniform));
		m_unsigned_uniform = nullptr;

		if (m_signed_uniform) CudaErrorCheck(cudaFree(m_signed_uniform));
		m_signed_uniform = nullptr;*/
	}

	void RandomNumbers::Reconstruct(cudaStream_t& mirror_stream)
	{
		float* hRandNumbers = (float*)s_hpm.GetPointerToMemory();

		// [>] Generate unsigned uniform random floats
		for (uint32_t i = 0; i < s_count; ++i)
			m_unsigned_uniform[i] = (rand() % RAND_MAX) / static_cast<float>(RAND_MAX);


		// [>] Generate random seeds
		for (uint32_t i = 0u; i < s_seeds_count; ++i)
			m_seeds[i] = rand() % s_seeds_count;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



	// ~~~~~~~~ [CLASS] CudaRenderingKErnel ~~~~~~~~
	CudaKernelData::CudaKernelData()
		: renderIndex(0u)
	{}
	CudaKernelData::~CudaKernelData()
	{}

	void CudaKernelData::Reconstruct(
		unsigned int renderIndex,
		cudaStream_t& mirrorStream)
	{
		this->renderIndex = renderIndex;
		this->randomNumbers.Reconstruct(mirrorStream);
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



	// ~~~~~~~~ [STRUCT] CudaTexture ~~~~~~~~
	cudaChannelFormatDesc CudaTexture::chanelDesc = cudaCreateChannelDesc<uchar4>();

	CudaTexture::CudaTexture()
		: textureArray(nullptr)
		, textureObject(0)
	{}
	CudaTexture::~CudaTexture()
	{
		if (textureObject) CudaErrorCheck(cudaDestroyTextureObject(textureObject));
		if (textureArray)  CudaErrorCheck(cudaFreeArray(textureArray));

		this->textureObject = 0;
		this->textureArray = nullptr;
	}

	void CudaTexture::Reconstruct(
		const Texture& host_texture, 
		cudaStream_t& mirror_stream)
	{
		if (this->textureArray == nullptr)
		{//--> hostMesh has texture but device equivalent doesn't

			// texture array allocation
			CudaErrorCheck(cudaMallocArray(
				&this->textureArray,
				&this->chanelDesc,
				host_texture.GetBitmap().GetWidth(), host_texture.GetBitmap().GetHeight()));

			// copy host texture data to device array
			CudaErrorCheck(cudaMemcpyToArray(
				this->textureArray,
				0, 0, host_texture.GetBitmap().GetMapAddress(),
				host_texture.GetBitmap().GetWidth() * host_texture.GetBitmap().GetHeight() * sizeof(Graphics::Color),
				cudaMemcpyKind::cudaMemcpyHostToDevice));

			// specify resource description			
			memset(&this->resDesc, 0, sizeof(cudaResourceDesc));
			this->resDesc.resType = cudaResourceType::cudaResourceTypeArray;
			this->resDesc.res.array.array = this->textureArray;

			// specify texture object parameters
			memset(&this->textureDesc, 0, (sizeof(cudaTextureDesc)));
			this->textureDesc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeWrap;
			this->textureDesc.addressMode[1] = cudaTextureAddressMode::cudaAddressModeWrap;
			if (host_texture.GetFilterMode() == Texture::FilterMode::Point)	
				this->textureDesc.filterMode = cudaTextureFilterMode::cudaFilterModePoint;
			else															
				this->textureDesc.filterMode = cudaTextureFilterMode::cudaFilterModeLinear;
			this->textureDesc.readMode = cudaTextureReadMode::cudaReadModeNormalizedFloat;
			this->textureDesc.normalizedCoords = 1;

			// craete texture object
			CudaErrorCheck(cudaCreateTextureObject(
				&this->textureObject,
				&this->resDesc,
				&this->textureDesc,
				nullptr));
		}
		else
		{//--> Both hostMesh and deviceMesh have texture

			// get texture array info (width and height)
			cudaExtent arrayInfo;
			CudaErrorCheck(cudaArrayGetInfo(nullptr, &arrayInfo, nullptr, this->textureArray));

			if (arrayInfo.width * arrayInfo.height != host_texture.GetBitmap().GetWidth() * host_texture.GetBitmap().GetHeight())
			{//--> size of hostMesh texture and CudaMesh texture doesn't match

				// free CudaMesh array
				CudaErrorCheck(cudaFreeArray(this->textureArray));

				// array allocation
				CudaErrorCheck(cudaMallocArray(
					&this->textureArray,
					&this->chanelDesc,
					host_texture.GetBitmap().GetWidth(), host_texture.GetBitmap().GetHeight()));
				this->resDesc.res.array.array = this->textureArray;

				// copy host texture data to device array
				CudaErrorCheck(cudaMemcpyToArray(
					this->textureArray,
					0, 0, host_texture.GetBitmap().GetMapAddress(),
					host_texture.GetBitmap().GetWidth() * host_texture.GetBitmap().GetHeight()* sizeof(Graphics::Color),
					cudaMemcpyKind::cudaMemcpyHostToDevice));
			}
			else
			{//--> Everything does match so do asynchronous texture update (TODO)

				//// copy host texture data to device array
				//CudaErrorCheck(cudaMemcpyToArray(
				//	this->textureArray,
				//	0, 0, host_texture.GetBitmap().GetMapAddress(),
				//	host_texture.GetBitmap().GetWidth() * host_texture.GetBitmap().GetHeight() * sizeof(Graphics::Color),
				//	cudaMemcpyKind::cudaMemcpyHostToDevice));

				// copy host texture data to device array
				CudaErrorCheck(cudaMemcpyToArrayAsync(
					this->textureArray,
					0, 0, host_texture.GetBitmap().GetMapAddress(),
					host_texture.GetBitmap().GetWidth() * host_texture.GetBitmap().GetHeight() * sizeof(Graphics::Color),
					cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
			}
		}
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



	// ~~~~~~~~ [STRUCT] CudaTriangle ~~~~~~~~
	CudaTriangle::CudaTriangle(const Triangle& hostTriangle)
		: v1(nullptr), v2(nullptr), v3(nullptr)
		, t1(nullptr), t2(nullptr), t3(nullptr)
		, n1(nullptr), n2(nullptr), n3(nullptr)
	{
		this->normal = hostTriangle.normal;
		this->color = hostTriangle.color;
	}
	CudaTriangle::~CudaTriangle()
	{
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}