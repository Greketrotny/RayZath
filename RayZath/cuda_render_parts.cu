#include "cuda_render_parts.cuh"
#include "rzexception.h"
#include "cuda_world.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		// ~~~~~~~~ [STRUCT] RNG ~~~~~~~~
		void RNG::Reconstruct()
		{
			// generate random numbers
			for (uint32_t i = 0u; i < s_count; ++i)
				m_unsigned_uniform[i] = (rand() % RAND_MAX) / static_cast<float>(RAND_MAX);
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		// ~~~~~~~~ [SRUCT] Seeds ~~~~~~~~
		void Seeds::Reconstruct(cudaStream_t& stream)
		{
			// generate random seeds
			for (uint32_t i = 0u; i < s_count; ++i)
				m_seeds[i] = rand() % s_count;
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



		// ~~~~~~~~ [STRUCT] CudaConstantKernel ~~~~~~~~
		void CudaConstantKernel::Reconstruct()
		{
			m_rng.Reconstruct();
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		// ~~~~~~~~ [CLASS] CudaGlobalKernel ~~~~~~~~
		CudaGlobalKernel::CudaGlobalKernel()
			: m_render_idx(0u)
		{}
		CudaGlobalKernel::~CudaGlobalKernel()
		{}

		void CudaGlobalKernel::Reconstruct(
			uint32_t render_idx,
			cudaStream_t& stream)
		{
			m_render_idx = render_idx;
			m_seeds.Reconstruct(stream);
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



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
			const CudaWorld& hCudaWorld,
			const Handle<Texture>& hTexture,
			cudaStream_t& mirror_stream)
		{
			if (!hTexture->GetStateRegister().IsModified()) return;

			if (this->textureArray == nullptr)
			{// hostMesh has texture but device equivalent doesn't

				// texture array allocation
				CudaErrorCheck(cudaMallocArray(
					&this->textureArray,
					&this->chanelDesc,
					hTexture->GetBitmap().GetWidth(), hTexture->GetBitmap().GetHeight()));

				// copy host texture data to device array
				CudaErrorCheck(cudaMemcpyToArray(
					this->textureArray,
					0u, 0u, hTexture->GetBitmap().GetMapAddress(),
					hTexture->GetBitmap().GetWidth() * hTexture->GetBitmap().GetHeight() * sizeof(Graphics::Color),
					cudaMemcpyKind::cudaMemcpyHostToDevice));

				// specify resource description			
				memset(&this->resDesc, 0, sizeof(cudaResourceDesc));
				this->resDesc.resType = cudaResourceType::cudaResourceTypeArray;
				this->resDesc.res.array.array = this->textureArray;

				// specify texture object parameters
				memset(&this->textureDesc, 0, (sizeof(cudaTextureDesc)));

				// address mode
				switch (hTexture->GetAddressMode())
				{
					case Texture::AddressMode::Wrap:
						this->textureDesc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeWrap;
						this->textureDesc.addressMode[1] = cudaTextureAddressMode::cudaAddressModeWrap;
						break;
					case Texture::AddressMode::Clamp:
						this->textureDesc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeClamp;
						this->textureDesc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeClamp;
						break;
					case Texture::AddressMode::Mirror:
						this->textureDesc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeMirror;
						this->textureDesc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeMirror;
						break;
					case Texture::AddressMode::Border:
						this->textureDesc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeBorder;
						this->textureDesc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeBorder;
						break;
				}

				// filter mode
				switch (hTexture->GetFilterMode())
				{
					case Texture::FilterMode::Point:
						this->textureDesc.filterMode = cudaTextureFilterMode::cudaFilterModePoint;
						break; 
					case Texture::FilterMode::Linear:
							this->textureDesc.filterMode = cudaTextureFilterMode::cudaFilterModeLinear;
							break;
				}

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
			{// Both hostMesh and deviceMesh have textures

				// get texture array info (width and height)
				cudaExtent arrayInfo;
				CudaErrorCheck(cudaArrayGetInfo(nullptr, &arrayInfo, nullptr, this->textureArray));

				if (arrayInfo.width * arrayInfo.height != 
					hTexture->GetBitmap().GetWidth() * hTexture->GetBitmap().GetHeight())
				{// size of hostMesh texture and CudaMesh texture don't match

					// free CudaMesh array
					CudaErrorCheck(cudaFreeArray(this->textureArray));

					// array allocation
					CudaErrorCheck(cudaMallocArray(
						&this->textureArray,
						&this->chanelDesc,
						hTexture->GetBitmap().GetWidth(), hTexture->GetBitmap().GetHeight()));
					this->resDesc.res.array.array = this->textureArray;

					// copy host texture data to device array
					CudaErrorCheck(cudaMemcpyToArray(
						this->textureArray,
						0u, 0u, hTexture->GetBitmap().GetMapAddress(),
						hTexture->GetBitmap().GetWidth() * hTexture->GetBitmap().GetHeight() * sizeof(Graphics::Color),
						cudaMemcpyKind::cudaMemcpyHostToDevice));
				}
				else
				{// Everything does match so do asynchronous texture update (TODO)

					// copy host texture data to device array
					CudaErrorCheck(cudaMemcpyToArrayAsync(
						this->textureArray,
						0u, 0u, hTexture->GetBitmap().GetMapAddress(),
						hTexture->GetBitmap().GetWidth() * 
						hTexture->GetBitmap().GetHeight() * 
						sizeof(*(hTexture->GetBitmap().GetMapAddress())),
						cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
					CudaErrorCheck(cudaStreamSynchronize(mirror_stream));
				}
			}

			hTexture->GetStateRegister().MakeUnmodified();
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



		// ~~~~~~~~ [STRUCT] CudaTriangle ~~~~~~~~
		CudaTriangle::CudaTriangle(const Triangle& hostTriangle)
			: v1(nullptr), v2(nullptr), v3(nullptr)
			, t1(nullptr), t2(nullptr), t3(nullptr)
			, n1(nullptr), n2(nullptr), n3(nullptr)
		{
			this->normal = hostTriangle.normal;
			this->material_id = hostTriangle.material_id & 0x3F;
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	}
}