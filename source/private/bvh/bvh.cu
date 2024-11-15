#include "../../public/bvh/bvh.h"
#include <algorithm>
#include <random>
#include "../../public/util.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "../../public/mathOperations.h"

constexpr auto MAX_TREE_HEIGHT = 64;

bool boxCompare(const bvhDataNode& a, const bvhDataNode& b, int axisIndex)
{
	interval aAxisInterval = a.bbox.axisInterval(axisIndex);
	interval bAxisInterval = b.bbox.axisInterval(axisIndex);

	return aAxisInterval.min < bAxisInterval.min;
}

bool boxXCompare(const bvhDataNode& a, const bvhDataNode& b)
{
	return boxCompare(a, b, 0);
}

bool boxYCompare(const bvhDataNode& a, const bvhDataNode& b)
{
	return boxCompare(a, b, 1);
}

bool boxZCompare(const bvhDataNode& a, const bvhDataNode& b)
{
	return boxCompare(a, b, 2);
}

struct _bvhNode
{
	aabb bbox;
	std::shared_ptr<_bvhNode> left;
	std::shared_ptr<_bvhNode> right;
	bvhDataNode data;

	_bvhNode() = default;

	_bvhNode(std::vector<bvhDataNode>& objects, size_t start, size_t end)
	{
		int axis = mathOperations::randomInt(0, 2);

		auto comparator = (axis == 0) ? boxXCompare
			: (axis == 1) ? boxYCompare
			: boxZCompare;

		size_t objectSpan = end - start;

		if (objectSpan == 1) 
		{
			data = objects[start];
			bbox = objects[start].bbox;
		}

		std::sort(objects.begin() + start, objects.end() + end, comparator);

		size_t middle = start + objectSpan / 2;

		left = std::make_shared<_bvhNode>(objects, start, middle);
		right = std::make_shared<_bvhNode>(objects, middle, end);

		bbox = aabb(left->bbox, right->bbox);
	}

	static std::vector<bvhNode> toLinearizedBvhNode(std::shared_ptr<_bvhNode> root, int& treeHeight)
	{
		treeHeight = 0;

		if (root == nullptr)
		{
			return {};
		}

		std::vector<bvhNode> nodes;
		int index = 0;

		std::vector<std::tuple<std::shared_ptr<_bvhNode>, int, int>> stack;
		stack.emplace_back(root, -1, 0);

		while (!stack.empty())
		{
			auto node = std::get<0>(stack.back());
			auto parentIndex = std::get<1>(stack.back());
			int depth = std::get<2>(stack.back());
			stack.pop_back();

			treeHeight = std::max(depth, treeHeight);

			nodes.emplace_back(node->data.obj, node->bbox);
			if (parentIndex != -1)
			{
				if (nodes[parentIndex].left == -1)
				{
					nodes[parentIndex].left = index;
				}
				else
				{
					nodes[parentIndex].right = index;
				}
			}

			if (node->left != nullptr)
			{
				stack.emplace_back(node->left, index, depth + 1);
			}
			if (node->right != nullptr)
			{
				stack.emplace_back(node->right, index, depth + 1);
			}

			index++;
		}

		return nodes;
	}
};

__host__ void bvhNode::allocateTree(bvhNode* nodes, int size)
{
	int treeSize = 2 * size;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&nodes), treeSize * sizeof(bvhNode)));
}

__host__ int bvhNode::buildTree(bvhNode* nodes, int size)
{
	std::vector<bvhDataNode> dataNodes(size);
	dataNodes.reserve(size);
	for (int i = 0; i < size; i++)
	{
		dataNodes.push_back(nodes[i]);
	}

	auto root = std::make_shared<_bvhNode>(dataNodes, 0, dataNodes.size());

	int treeHeight = 0;
	std::vector<bvhNode> linearizedNodes = _bvhNode::toLinearizedBvhNode(root, treeHeight);

	if (treeHeight > MAX_TREE_HEIGHT)
	{
		throw std::runtime_error("Tree height exceeds maximum tree height");
	}

	for (int i = 0; i < linearizedNodes.size(); i++)
	{
		nodes[i] = linearizedNodes[i];
	}

	return treeHeight;
}

__device__ bool bvhNode::checkIntersection(const bvhNode* nodes, Ray& ray, interval hitRange, HitInformation& hitInformation)
{
	int stack[MAX_TREE_HEIGHT];
	int stackPtr = 0;

	stack[stackPtr++] = 0;

	bool hitAnything = false;

	while (stackPtr > 0)
	{
		int nodeIdx = stack[--stackPtr];
		const bvhNode& node = nodes[nodeIdx];

		if (node.bbox.checkIntersection(ray, hitRange))
		{
			if (node.left == -1 && node.right == -1)
			{
				if (node.obj->checkIntersection(ray, hitRange, hitInformation))
				{
					hitAnything = true;
					hitRange = interval(hitRange.min, hitInformation.distance);
				}
			}
			else
			{
				if (node.left != -1)
					stack[stackPtr++] = node.left;

				if (node.right != -1)
					stack[stackPtr++] = node.right;
			}
		}
	}

	return hitAnything;
}

__device__ void bvhNode::prefillNodes(bvhNode* nodes, Shape** shapes, int listSize)
{
	for (int i = 0; i < listSize; i++)
	{
		nodes[i] = bvhNode(shapes[i]);
	}
}