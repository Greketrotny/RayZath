#ifndef CUDA_MESH_CUH
#define CUDA_MESH_CUH

#include "mesh.h"
#include "cuda_render_object.cuh"
#include "cuda_render_parts.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		typedef cudaVec3<float> CudaVertex;
		typedef cudaVec3<float> CudaNormal;
		template <class HostComponent, class CudaComponent>

		struct CudaComponentContainer
		{
		private:
			CudaComponent* memory;
			uint32_t capacity, count;


		public:
			__host__ CudaComponentContainer()
				: memory(nullptr)
				, capacity(0u)
				, count(0u)
			{}
			__host__ ~CudaComponentContainer()
			{
				if (memory) CudaErrorCheck(cudaFree(memory));
				memory = nullptr;

				capacity = 0u;
				count = 0u;
			}


		public:
			__host__ void Reconstruct(
				const ComponentContainer<HostComponent>& hComponents,
				HostPinnedMemory& hpm,
				cudaStream_t& mirror_stream)
			{
				count = hComponents.GetCount();

				if (hComponents.GetCapacity() != capacity)
				{//--> capacities don't match

					// free memory
					if (this->memory) CudaErrorCheck(cudaFree(memory));

					// update count and capacity
					this->count = hComponents.GetCount();
					this->capacity = hComponents.GetCapacity();

					// allocate new memory
					CudaErrorCheck(cudaMalloc(&memory, capacity * sizeof(CudaComponent)));

					// copy data from hostMesh to cudaMesh
					CudaComponent* hCudaComponents = (CudaComponent*)malloc(count * sizeof(CudaComponent));
					for (uint32_t i = 0u; i < count; ++i)
					{
						new (&hCudaComponents[i]) CudaComponent(hComponents[i]);
					}
					CudaErrorCheck(cudaMemcpy(
						memory, hCudaComponents,
						count * sizeof(CudaComponent),
						cudaMemcpyKind::cudaMemcpyHostToDevice));
					free(hCudaComponents);
				}
				else
				{// capacities match so perform asnynchronous copying

					// divide work into chunks of components to fit in host pinned memory
					uint32_t chunkSize = hpm.GetSize() / sizeof(*memory);
					if (chunkSize == 0u) return;	// TODO: throw exception (too few memory for async copying)

					// reconstruct each component
					for (uint32_t startIndex = 0u; startIndex < count; startIndex += chunkSize)
					{
						if (startIndex + chunkSize > count) chunkSize = count - startIndex;

						// copy from device memory
						CudaComponent* const hCudaComponents = (CudaComponent*)hpm.GetPointerToMemory();
						CudaErrorCheck(cudaMemcpyAsync(
							hCudaComponents, memory + startIndex,
							chunkSize * sizeof(CudaComponent),
							cudaMemcpyKind::cudaMemcpyDeviceToHost, mirror_stream));
						CudaErrorCheck(cudaStreamSynchronize(mirror_stream));

						// loop through all components in the chunk
						for (uint32_t i = 0u; i < chunkSize; ++i)
						{
							new (&hCudaComponents[i]) CudaComponent(hComponents[startIndex + i]);
						}

						// copy mirrored components back to device
						CudaErrorCheck(cudaMemcpyAsync(
							memory + startIndex, hCudaComponents,
							chunkSize * sizeof(*memory),
							cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
						CudaErrorCheck(cudaStreamSynchronize(mirror_stream));
					}
				}
			}

		public:
			__host__ __device__ __inline__ CudaComponent& operator[](const uint32_t& index)
			{
				return memory[index];
			}
			__host__ __device__ __inline__ const CudaComponent& operator[](const uint32_t& index) const
			{
				return memory[index];
			}
		public:
			__host__ const CudaComponent* GetMemoryAddress() const
			{
				return memory;
			}
			__host__ CudaComponent* GetMemoryAddress()
			{
				return memory;
			}
			__host__ __device__ __inline__ const uint32_t& GetCapacity() const
			{
				return capacity;
			}
			__host__ __device__ __inline__ const uint32_t& GetCount() const
			{
				return count;
			}
		};
		template<> struct CudaComponentContainer<Triangle, CudaTriangle>
		{
		private:
			CudaTriangle* memory;
			uint32_t capacity, count;


		public:
			__host__ CudaComponentContainer()
				: memory(nullptr)
				, capacity(0u)
				, count(0u)
			{}
			__host__ ~CudaComponentContainer()
			{
				if (memory) CudaErrorCheck(cudaFree(memory));
				memory = nullptr;

				capacity = 0u;
				count = 0u;
			}


		public:
			__host__ void Reconstruct(
				const Handle<MeshStructure>& hMeshStructure,
				CudaComponentContainer<Vertex, CudaVertex>& hCudaVertices,
				CudaComponentContainer<Texcrd, CudaTexcrd>& hCudaTexcrds,
				CudaComponentContainer<Normal, CudaNormal>& hCudaNormals,
				HostPinnedMemory& hpm,
				cudaStream_t& mirror_stream)
			{
				count = hMeshStructure->GetTriangles().GetCount();

				if (hMeshStructure->GetTriangles().GetCapacity() != capacity)
				{//--> capacities don't match

					// free memory
					if (this->memory) CudaErrorCheck(cudaFree(memory));

					// update count and capacity
					this->count = hMeshStructure->GetTriangles().GetCount();
					this->capacity = hMeshStructure->GetTriangles().GetCapacity();

					// allocate new memory
					CudaErrorCheck(cudaMalloc(&memory, capacity * sizeof(CudaTriangle)));

					// copy data from hostMesh to cudaMesh
					CudaTriangle* hCudaTriangles = (CudaTriangle*)malloc(count * sizeof(CudaTriangle));
					for (uint32_t i = 0u; i < count; ++i)
					{
						new (&hCudaTriangles[i]) CudaTriangle(hMeshStructure->GetTriangles()[i]);

						if (hMeshStructure->GetTriangles()[i].v1 != nullptr) hCudaTriangles[i].v1 = &hCudaVertices[uint32_t(hMeshStructure->GetTriangles()[i].v1 - &hMeshStructure->GetVertices()[0])];
						if (hMeshStructure->GetTriangles()[i].v2 != nullptr) hCudaTriangles[i].v2 = &hCudaVertices[uint32_t(hMeshStructure->GetTriangles()[i].v2 - &hMeshStructure->GetVertices()[0])];
						if (hMeshStructure->GetTriangles()[i].v3 != nullptr) hCudaTriangles[i].v3 = &hCudaVertices[uint32_t(hMeshStructure->GetTriangles()[i].v3 - &hMeshStructure->GetVertices()[0])];

						if (hMeshStructure->GetTriangles()[i].t1 != nullptr) hCudaTriangles[i].t1 = &hCudaTexcrds[uint32_t(hMeshStructure->GetTriangles()[i].t1 - &hMeshStructure->GetTexcrds()[0])];
						if (hMeshStructure->GetTriangles()[i].t2 != nullptr) hCudaTriangles[i].t2 = &hCudaTexcrds[uint32_t(hMeshStructure->GetTriangles()[i].t2 - &hMeshStructure->GetTexcrds()[0])];
						if (hMeshStructure->GetTriangles()[i].t3 != nullptr) hCudaTriangles[i].t3 = &hCudaTexcrds[uint32_t(hMeshStructure->GetTriangles()[i].t3 - &hMeshStructure->GetTexcrds()[0])];

						if (hMeshStructure->GetTriangles()[i].n1 != nullptr) hCudaTriangles[i].n1 = &hCudaNormals[uint32_t(hMeshStructure->GetTriangles()[i].n1 - &hMeshStructure->GetNormals()[0])];
						if (hMeshStructure->GetTriangles()[i].n2 != nullptr) hCudaTriangles[i].n2 = &hCudaNormals[uint32_t(hMeshStructure->GetTriangles()[i].n2 - &hMeshStructure->GetNormals()[0])];
						if (hMeshStructure->GetTriangles()[i].n3 != nullptr) hCudaTriangles[i].n3 = &hCudaNormals[uint32_t(hMeshStructure->GetTriangles()[i].n3 - &hMeshStructure->GetNormals()[0])];
					}
					CudaErrorCheck(cudaMemcpy(
						memory, hCudaTriangles,
						count * sizeof(CudaTriangle),
						cudaMemcpyKind::cudaMemcpyHostToDevice));
					free(hCudaTriangles);
				}
				else
				{// capacities match so perform asnynchronous copying

					// divide work into chunks of components to fit in host pinned memory
					uint32_t chunkSize = hpm.GetSize() / sizeof(*memory);
					if (chunkSize == 0u) return;	// TODO: throw exception (too few memory for async copying)

					// reconstruct each component
					for (uint32_t startIndex = 0u; startIndex < count; startIndex += chunkSize)
					{
						if (startIndex + chunkSize > count) chunkSize = count - startIndex;

						// copy from device memory
						CudaTriangle* const hCudaTriangles = (CudaTriangle*)hpm.GetPointerToMemory();
						CudaErrorCheck(cudaMemcpyAsync(
							hCudaTriangles, memory + startIndex,
							chunkSize * sizeof(CudaTriangle),
							cudaMemcpyKind::cudaMemcpyDeviceToHost, mirror_stream));
						CudaErrorCheck(cudaStreamSynchronize(mirror_stream));

						// loop through all components in the chunk
						for (uint32_t i = 0u; i < chunkSize; ++i)
						{
							new (&hCudaTriangles[i]) CudaTriangle(hMeshStructure->GetTriangles()[startIndex + i]);

							if (hMeshStructure->GetTriangles()[i].v1 != nullptr) hCudaTriangles[i].v1 = &hCudaVertices[uint32_t(hMeshStructure->GetTriangles()[startIndex + i].v1 - &hMeshStructure->GetVertices()[0])];
							if (hMeshStructure->GetTriangles()[i].v2 != nullptr) hCudaTriangles[i].v2 = &hCudaVertices[uint32_t(hMeshStructure->GetTriangles()[startIndex + i].v2 - &hMeshStructure->GetVertices()[0])];
							if (hMeshStructure->GetTriangles()[i].v3 != nullptr) hCudaTriangles[i].v3 = &hCudaVertices[uint32_t(hMeshStructure->GetTriangles()[startIndex + i].v3 - &hMeshStructure->GetVertices()[0])];

							if (hMeshStructure->GetTriangles()[i].t1 != nullptr) hCudaTriangles[i].t1 = &hCudaTexcrds[uint32_t(hMeshStructure->GetTriangles()[startIndex + i].t1 - &hMeshStructure->GetTexcrds()[0])];
							if (hMeshStructure->GetTriangles()[i].t2 != nullptr) hCudaTriangles[i].t2 = &hCudaTexcrds[uint32_t(hMeshStructure->GetTriangles()[startIndex + i].t2 - &hMeshStructure->GetTexcrds()[0])];
							if (hMeshStructure->GetTriangles()[i].t3 != nullptr) hCudaTriangles[i].t3 = &hCudaTexcrds[uint32_t(hMeshStructure->GetTriangles()[startIndex + i].t3 - &hMeshStructure->GetTexcrds()[0])];

							if (hMeshStructure->GetTriangles()[i].n1 != nullptr) hCudaTriangles[i].n1 = &hCudaNormals[uint32_t(hMeshStructure->GetTriangles()[startIndex + i].n1 - &hMeshStructure->GetNormals()[0])];
							if (hMeshStructure->GetTriangles()[i].n2 != nullptr) hCudaTriangles[i].n2 = &hCudaNormals[uint32_t(hMeshStructure->GetTriangles()[startIndex + i].n2 - &hMeshStructure->GetNormals()[0])];
							if (hMeshStructure->GetTriangles()[i].n3 != nullptr) hCudaTriangles[i].n3 = &hCudaNormals[uint32_t(hMeshStructure->GetTriangles()[startIndex + i].n3 - &hMeshStructure->GetNormals()[0])];
						}

						// copy mirrored components back to device
						CudaErrorCheck(cudaMemcpyAsync(
							memory + startIndex, hCudaTriangles,
							chunkSize * sizeof(*memory),
							cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
						CudaErrorCheck(cudaStreamSynchronize(mirror_stream));
					}
				}
			}

		public:
			__host__ __device__ __inline__ CudaTriangle& operator[](const uint32_t& index)
			{
				return memory[index];
			}
			__host__ __device__ __inline__ const CudaTriangle& operator[](const uint32_t& index) const
			{
				return memory[index];
			}

		public:
			__host__ const CudaTriangle* GetMemoryAddress() const
			{
				return memory;
			}
			__host__ CudaTriangle* GetMemoryAddress()
			{
				return memory;
			}
			__host__ __device__ __inline__ const uint32_t& GetCapacity() const
			{
				return capacity;
			}
			__host__ __device__ __inline__ const uint32_t& GetCount() const
			{
				return count;
			}
		};


		struct CudaComponentTreeNode
		{
		public:
			CudaComponentTreeNode* m_child[8];
			bool m_is_leaf;
			uint32_t m_leaf_first_index, m_leaf_last_index;
			CudaBoundingBox m_bb;


		public:
			__host__ CudaComponentTreeNode()
				: m_is_leaf(true)
				, m_leaf_first_index(0u)
				, m_leaf_last_index(0u)
			{
				for (int i = 0; i < 8; i++) m_child[i] = nullptr;
			}
			template <class HostObject>
			__host__ CudaComponentTreeNode(const ComponentTreeNode<HostObject>& hNode)
				: m_is_leaf(hNode.IsLeaf())
				, m_leaf_first_index(0u)
				, m_leaf_last_index(0u)
				, m_bb(hNode.GetBoundingBox())
			{
				for (int i = 0; i < 8; i++) m_child[i] = nullptr;
			}
		};
		template <class HostComponent, class CudaComponent>
		class CudaComponentBVH
		{
		private:
			CudaComponentTreeNode* m_nodes;
			uint32_t m_nodes_capacity, m_nodes_count;
			CudaComponentTreeNode* mp_traversable_node;

			CudaComponent** m_ptrs;
			uint32_t m_ptrs_capacity, m_ptrs_count;


		public:
			__host__ CudaComponentBVH()
				: m_nodes(nullptr)
				, m_nodes_capacity(0u)
				, m_nodes_count(0u)
				, m_ptrs(nullptr)
				, m_ptrs_capacity(0u)
				, m_ptrs_count(0u)
				, mp_traversable_node(nullptr)
			{
				// create traversable node
				CudaComponentTreeNode hTraversable;
				CudaErrorCheck(cudaMalloc(&mp_traversable_node, sizeof(*mp_traversable_node)));
				CudaErrorCheck(cudaMemcpy(mp_traversable_node, &hTraversable,
					sizeof(*mp_traversable_node),
					cudaMemcpyKind::cudaMemcpyHostToDevice));
			}
			__host__ ~CudaComponentBVH()
			{
				// delete tree nodes
				if (m_nodes) CudaErrorCheck(cudaFree(m_nodes));
				m_nodes = nullptr;
				m_nodes_capacity = 0u;
				m_nodes_count = 0u;

				// delete objects pointers
				if (m_ptrs) CudaErrorCheck(cudaFree(m_ptrs));
				m_ptrs = nullptr;
				m_ptrs_capacity = 0u;
				m_ptrs_count = 0u;

				// delete traversable node
				if (mp_traversable_node) CudaErrorCheck(cudaFree(mp_traversable_node));
				mp_traversable_node = nullptr;
			}


		public:
			__host__ void Reconstruct(
				ComponentContainer<HostComponent, true>& hContainer,
				CudaComponentContainer<HostComponent, CudaComponent>& hCudaContainer,
				HostPinnedMemory& hpm,
				cudaStream_t& mirror_stream)
			{
				//if (hContainer.GetBVH().GetRootNode() == nullptr) return;	// host bvh is empty

				uint32_t h_tree_size = hContainer.GetBVH().GetTreeSize();

				// [>] Resize capacities
				// resize nodes storage capacity
				if (m_nodes_capacity != h_tree_size)
				{
					m_nodes_capacity = h_tree_size;
					if (m_nodes) CudaErrorCheck(cudaFree(m_nodes));
					CudaErrorCheck(cudaMalloc((void**)&m_nodes, h_tree_size * sizeof(*m_nodes)));
				}

				// resize ptrs storage capacity
				if (m_ptrs_capacity != hContainer.GetCapacity())
				{
					m_ptrs_capacity = hContainer.GetCapacity();
					if (m_ptrs) CudaErrorCheck(cudaFree(m_ptrs));
					CudaErrorCheck(cudaMalloc((void**)&m_ptrs, m_ptrs_capacity * sizeof(*m_ptrs)));
				}

				if (m_ptrs_capacity == 0u || m_nodes_capacity == 0u) return;


				// [>] Allocate host memory
				CudaComponentTreeNode* hCudaTreeNodes =
					(CudaComponentTreeNode*)malloc(m_nodes_capacity * sizeof(*hCudaTreeNodes));
				CudaComponent** hCudaObjectPtrs =
					(CudaComponent**)malloc(m_ptrs_capacity * sizeof(*hCudaObjectPtrs));

				m_nodes_count = 0u;
				m_ptrs_count = 0u;


				// [>] Construct BVH
				new (&hCudaTreeNodes[m_nodes_count]) CudaComponentTreeNode(hContainer.GetBVH().GetRootNode());
				++m_nodes_count;
				FillNode(
					hCudaTreeNodes + m_nodes_count - 1u,
					hContainer.GetBVH().GetRootNode(),
					hCudaContainer,
					hContainer,
					hCudaTreeNodes, hCudaObjectPtrs);


				// [>] Copy memory to device
				// copy tree nodes
				CudaErrorCheck(cudaMemcpy(
					m_nodes, hCudaTreeNodes,
					m_nodes_capacity * sizeof(CudaComponentTreeNode),
					cudaMemcpyKind::cudaMemcpyHostToDevice));
				// copy object pointers
				CudaErrorCheck(cudaMemcpy(
					m_ptrs, hCudaObjectPtrs,
					m_ptrs_capacity * sizeof(CudaComponent*),
					cudaMemcpyKind::cudaMemcpyHostToDevice));


				// [>] Free host memory
				free(hCudaTreeNodes);
				free(hCudaObjectPtrs);
			}
		private:
			__host__ uint32_t CreateLeaf(uint32_t size)
			{
				if (m_ptrs_count + size > m_ptrs_capacity) return 0u;
				else
				{
					m_ptrs_count += size;
					return m_ptrs_count - size;
				}
			}
			__host__ void FillNode(
				CudaComponentTreeNode* hCudaNode,
				const ComponentTreeNode<HostComponent>& hNode,
				CudaComponentContainer<HostComponent, CudaComponent>& hCudaContainer,
				ComponentContainer<HostComponent>& hContainer,
				CudaComponentTreeNode* hCudaTreeNodes,
				CudaComponent** hCudaObjectPtrs)
			{
				if (hNode.IsLeaf())
				{
					uint32_t leaf_size = hNode.GetObjectCount();
					hCudaNode->m_leaf_first_index = CreateLeaf(leaf_size);
					hCudaNode->m_leaf_last_index = hCudaNode->m_leaf_first_index + leaf_size;
					for (uint32_t i = 0u; i < leaf_size; i++)
					{
						hCudaObjectPtrs[hCudaNode->m_leaf_first_index + i] =
							hCudaContainer.GetMemoryAddress() +
							(hNode.GetObject(i) - &hContainer[0]);
					}

					for (uint32_t i = 0u; i < 8u; i++)
					{
						//hCudaNode->m_child[i] = mp_traversable_node;
					}
				}
				else
				{
					for (int i = 0; i < 8; i++)
					{
						const ComponentTreeNode<HostComponent>* hChildNode = hNode.GetChild(i);
						if (hChildNode)
						{
							new (&hCudaTreeNodes[m_nodes_count]) CudaComponentTreeNode(*hChildNode);
							++m_nodes_count;

							hCudaNode->m_child[i] = m_nodes + (m_nodes_count - 1u);
							FillNode(
								hCudaTreeNodes + m_nodes_count - 1u,
								*hChildNode,
								hCudaContainer,
								hContainer,
								hCudaTreeNodes, hCudaObjectPtrs);
						}
						else
						{
							//hCudaNode->m_child[i] = mp_traversable_node;
						}
					}
				}
			}


		public:
			__device__ __inline__ void ClosestIntersection(
				TriangleIntersection& intersection) const
			{
				if (m_nodes_count == 0u) return;	// the tree is empty
				if (!m_nodes[0].m_bb.RayIntersection(intersection.ray)) return;	// ray misses root node

				CudaComponentTreeNode* node[16u];	// nodes in stack
				node[0] = &m_nodes[0];
				int8_t depth = 0;	// current depth
				// start node index (depends on ray direction)
				uint8_t start_node =
					(uint32_t(intersection.ray.direction.x > 0.0f) << 2ull) |
					(uint32_t(intersection.ray.direction.y > 0.0f) << 1ull) |
					(uint32_t(intersection.ray.direction.z > 0.0f));
				uint64_t child_counters = 0u;	// child counters mask (16 frames by 4 bits)


				while (depth >= 0 && depth < 15)
				{
					if (node[depth]->m_is_leaf)
					{
						// check all objects held by the node
						for (uint32_t i = node[depth]->m_leaf_first_index;
							i < node[depth]->m_leaf_last_index;
							i++)
						{
							m_ptrs[i]->RayIntersect(intersection);
						}
						--depth;
						continue;
					}

					// check checked children count
					if (((child_counters >> (4ull * depth)) & 0b1111ull) >= 8ull)
					{	// all children checked - decrement depth

						--depth;
						continue;
					}


					// get next child to check
					CudaComponentTreeNode* child_node =
						node[depth]->m_child[((child_counters >> (4ull * depth)) & 0b111ull) ^ start_node];
					// increment checked children count
					child_counters += (1ull << (4ull * depth));

					if (child_node)
					{
						if (child_node->m_bb.RayIntersection(intersection.ray))
						{
							intersection.bvh_factor *= (1.0f -
								0.01f * float(((child_counters >> (4ull * depth)) & 0b1111ull)));

							// increment depth
							++depth;
							// set current node to its child
							node[depth] = child_node;
							// clear checked children counter
							child_counters &= (~(0b1111ull << (4ull * uint64_t(depth))));
						}
					}
				}
			}
			__device__ __inline__ float AnyIntersection(
				TriangleIntersection& intersection) const
			{
				if (m_nodes_count == 0u) return 1.0f;	// the tree is empty
				if (!m_nodes[0].m_bb.RayIntersection(intersection.ray)) return 1.0f;	// ray misses root node

				CudaComponentTreeNode* node[16u];	// nodes in stack
				node[0] = &m_nodes[0];
				int8_t depth = 0;	// current depth
				// start node index (depends on ray direction)
				uint8_t start_node =
					(uint32_t(intersection.ray.direction.x > 0.0f) << 2ull) |
					(uint32_t(intersection.ray.direction.y > 0.0f) << 1ull) |
					(uint32_t(intersection.ray.direction.z > 0.0f));
				uint64_t child_counters = 0u;	// child counters mask (8 frames by 4 bits)

				float shadow = 1.0f;

				while (depth >= 0)
				{
					if (node[depth]->m_is_leaf)
					{
						// check all objects held by the node
						for (uint32_t i = node[depth]->m_leaf_first_index;
							i < node[depth]->m_leaf_last_index;
							i++)
						{
							if (m_ptrs[i]->RayIntersect(intersection))
							{
								return 0.0f;
								/*const CudaColor<float> color = mesh->FetchTextureWithUV(
									m_ptrs[i],
									intersection.b1,
									intersection.b2);
									shadow *= (1.0f - color.alpha);
									if (shadow < 0.0001f) return shadow;*/
							}
						}
						--depth;
					}
					else
					{
						if (depth > 15) return 1.0f;

						// check checked child count
						if (((child_counters >> (4ull * depth)) & 0b1111ull) >= 8ull)
						{	// all children checked - decrement depth
							--depth;
						}
						else
						{
							// get next child to check
							CudaComponentTreeNode* child_node =
								node[depth]->m_child[((child_counters >> (4ull * depth)) & 0b111ull) ^ start_node];
							// increment checked child count
							child_counters += (1ull << (4ull * depth));

							if (child_node)
							{
								if (child_node->m_bb.RayIntersection(intersection.ray))
								{
									intersection.bvh_factor *= (1.0f -
										0.01f * float(((child_counters >> (4ull * depth)) & 0b1111ull)));

									// increment depth
									++depth;
									// set current node to its child
									node[depth] = child_node;
									// clear checked child counter
									child_counters &= (~(0b1111ull << (4ull * uint64_t(depth))));
								}
							}
						}
					}
				}

				return shadow;
			}
		};

		template <class HostObject, class CudaObject>
		class CudaComponentContainerWithBVH;

		template <class Triangle, class CudaTriangle>
		class CudaComponentContainerWithBVH
		{
		private:
			CudaComponentContainer<Triangle, CudaTriangle> m_container;
			CudaComponentBVH<Triangle, CudaTriangle> m_bvh;


		public:
			__host__ CudaComponentContainerWithBVH()
			{

			}
			__host__ ~CudaComponentContainerWithBVH()
			{

			}


		public:
			__host__ void Reconstruct(
				const Handle<MeshStructure>& hMeshStructure,
				CudaComponentContainer<Vertex, CudaVertex>& hCudaVertices,
				CudaComponentContainer<Texcrd, CudaTexcrd>& hCudaTexcrds,
				CudaComponentContainer<Normal, CudaNormal>& hCudaNormals,
				HostPinnedMemory& hpm,
				cudaStream_t& mirror_stream)
			{
				//if (!hContainer.GetStateRegister().IsModified()) return;

				m_container.Reconstruct(
					hMeshStructure,
					hCudaVertices,
					hCudaTexcrds,
					hCudaNormals,
					hpm, mirror_stream);
				m_bvh.Reconstruct(
					hMeshStructure->GetTriangles(),
					m_container,
					hpm, mirror_stream);

				//hContainer.GetStateRegister().MakeUnmodified();
			}

			__device__ __inline__ const CudaComponentContainer<Triangle, CudaTriangle>& GetContainer() const
			{
				return m_container;
			}
			__device__ __inline__ const CudaComponentBVH<Triangle, CudaTriangle>& GetBVH() const
			{
				return m_bvh;
			}
		};


		struct CudaMeshStructure : public WithExistFlag
		{
		private:
			CudaComponentContainer<Vertex, CudaVertex> m_vertices;
			CudaComponentContainer<Texcrd, CudaTexcrd> m_texcrds;
			CudaComponentContainer<Math::vec3<float>, cudaVec3<float>> m_normals;
			CudaComponentContainerWithBVH<Triangle, CudaTriangle> m_triangles;

			static HostPinnedMemory hostPinnedMemory;

		public:
			__host__ CudaMeshStructure();
			__host__ ~CudaMeshStructure();


		public:
			__host__ void Reconstruct(
				const CudaWorld& hCudaWorld,
				const Handle<MeshStructure>& hMeshStructure,
				cudaStream_t& mirror_stream);
		public:
			__device__ __inline__ const CudaComponentContainer<Vertex, CudaVertex>& GetVertices() const
			{
				return m_vertices;
			}
			__device__ __inline__ const CudaComponentContainer<Texcrd, CudaTexcrd>& GetTexcrds() const
			{
				return m_texcrds;
			}
			__device__ __inline__ const CudaComponentContainer<Math::vec3<float>, cudaVec3<float>>& GetNormals() const
			{
				return m_normals;
			}
			__device__ __inline__ const CudaComponentContainerWithBVH<Triangle, CudaTriangle>& GetTriangles() const
			{
				return m_triangles;
			}
		};



		class CudaMesh : public CudaRenderObject
		{
		public:
			const CudaMeshStructure* mesh_structure;
			const CudaMaterial* materials[Mesh::GetMaterialCount()];


		public:
			__host__ CudaMesh();


		public:
			__host__ void Reconstruct(
				const CudaWorld& hCudaWorld, 
				const Handle<Mesh>& hMesh, 
				cudaStream_t& mirror_stream);


			// device rendering functions
		public:
			__device__ __inline__ bool RayIntersect(RayIntersection& intersection) const
			{
				// [>] check ray intersection with bounding_box
				if (!bounding_box.RayIntersection(intersection.ray))
					return false;

				// [>] transpose objectSpaceRay
				TriangleIntersection local_intersect;
				local_intersect.ray = intersection.ray;
				local_intersect.ray.origin -= this->position;
				local_intersect.ray.origin.RotateZYX(-rotation);
				local_intersect.ray.direction.RotateZYX(-rotation);
				local_intersect.ray.origin /= this->scale;
				local_intersect.ray.direction /= this->scale;
				local_intersect.ray.origin -= this->center;
				const float length_factor = local_intersect.ray.direction.Length();
				local_intersect.ray.length *= length_factor;
				local_intersect.ray.direction.Normalize();


				// Linear search
				/*for (uint32_t index = 0u;
					index < mesh_structure.GetTriangles().GetContainer().GetCount();
					++index)
				{
					const CudaTriangle* triangle = &mesh_structure.GetTriangles().GetContainer()[index];
					triangle->RayIntersect(local_intersect);
				}*/
				// BVH search
				if (mesh_structure == nullptr) return false;
				mesh_structure->GetTriangles().GetBVH().ClosestIntersection(local_intersect);


				// ~~~~ BVH debug
				intersection.bvh_factor *= local_intersect.bvh_factor;
				// ~~~~ BVH debug


				if (local_intersect.triangle)
				{
					intersection.texcrd = 
						local_intersect.triangle->TexcrdFromBarycenter(
						local_intersect.b1, local_intersect.b2);

					intersection.ray.length = local_intersect.ray.length / length_factor;

					// calculate mapped normal
					cudaVec3<float> mapped_normal;
					if (local_intersect.triangle->n1 &&
						local_intersect.triangle->n2 &&
						local_intersect.triangle->n3)
					{
						mapped_normal =
							(*local_intersect.triangle->n1 * (1.0f - local_intersect.b1 - local_intersect.b2) +
								*local_intersect.triangle->n2 * local_intersect.b1 +
								*local_intersect.triangle->n3 * local_intersect.b2);
					}
					else
					{
						mapped_normal = local_intersect.triangle->normal;
					}


					// reverse normal if looking at back side of triangle
					const bool reverse = cudaVec3<float>::DotProduct(
						local_intersect.triangle->normal,
						local_intersect.ray.direction) < 0.0f;
					const float reverse_factor = static_cast<float>(reverse) * 2.0f - 1.0f;


					// fill intersection structure
					intersection.surface_normal = local_intersect.triangle->normal * reverse_factor;
					intersection.mapped_normal = mapped_normal * reverse_factor;
					if (cudaVec3<float>::DotProduct(intersection.mapped_normal, local_intersect.ray.direction) > 0.0f)
						intersection.mapped_normal = intersection.surface_normal;

					intersection.surface_material = materials[local_intersect.triangle->material_id];

					// set material
					if (!reverse)
					{	// intersection from inside

						intersection.behind_material = nullptr;
					}
					else
					{	// intersection from outside

						intersection.behind_material = 
							intersection.surface_material;
					}

					return true;
				}

				return false;
			}
			__device__ __inline__ float ShadowRayIntersect(const CudaRay& ray) const
			{
				// [>] check ray intersection with bounding_box
				if (!bounding_box.RayIntersection(ray))
					return 1.0f;

				// [>] transpose objectSpaceRay
				CudaRay objectSpaceRay = ray;
				objectSpaceRay.origin -= this->position;
				objectSpaceRay.origin.RotateZYX(-rotation);
				objectSpaceRay.direction.RotateZYX(-rotation);
				objectSpaceRay.origin /= this->scale;
				objectSpaceRay.direction /= this->scale;
				objectSpaceRay.origin -= this->center;
				objectSpaceRay.length *= objectSpaceRay.direction.Length();
				objectSpaceRay.direction.Normalize();

				TriangleIntersection tri_intersection;
				tri_intersection.ray = objectSpaceRay;

				//float shadow = this->material.transmittance;
				if (mesh_structure == nullptr) return false;
				return mesh_structure->GetTriangles().GetBVH().AnyIntersection(tri_intersection);
			}
		private:
			__device__ __inline__ bool RayTriangleIntersect(
				const CudaRay& ray,
				const CudaTriangle* triangle,
				cudaVec3<float>& P,
				float& currDistance,
				const float& maxDistance) const
			{
				// check triangle normal - ray direction similarity
				float triangleFacing = cudaVec3<float>::DotProduct(triangle->normal, ray.direction);
				if (triangleFacing < 0.001f && triangleFacing > -0.001f)
					return false;

				// check triangle position
				float D = cudaVec3<float>::DotProduct(triangle->normal, *triangle->v1);
				float T = -(cudaVec3<float>::DotProduct(triangle->normal, ray.origin) - D) / triangleFacing;
				if (T <= 0.0f)
					return false;

				// find point of intersection with triangle's plane
				P = ray.origin + ray.direction * T;

				// check if found point is closer or further from ray.origin by required minPointDistance
				currDistance = (P - ray.origin).Length();
				if (currDistance > maxDistance)
					return false;

				// check if ray intersect triangle
				cudaVec3<float> C;
				cudaVec3<float> edge;
				cudaVec3<float> vp;

				edge = *triangle->v2 - *triangle->v1;
				vp = P - *triangle->v1;
				C = cudaVec3<float>::CrossProduct(edge, vp);
				if (cudaVec3<float>::DotProduct(triangle->normal, C) < 0.0f) return false;

				edge = *triangle->v3 - *triangle->v2;
				vp = P - *triangle->v2;
				C = cudaVec3<float>::CrossProduct(edge, vp);
				if (cudaVec3<float>::DotProduct(triangle->normal, C) < 0.0f) return false;

				edge = *triangle->v1 - *triangle->v3;
				vp = P - *triangle->v3;
				C = cudaVec3<float>::CrossProduct(edge, vp);
				if (cudaVec3<float>::DotProduct(triangle->normal, C) < 0.0f) return false;

				return true;
			}
		};
	}
}

#endif // !CUDA_MESH_CUH