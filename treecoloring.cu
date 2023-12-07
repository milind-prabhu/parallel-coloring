#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <string>

using namespace std;

__host__
void writeEdgesToFile(const string& filename, int numNodes) {
	ofstream outFile(filename);

	if (outFile.is_open()) {
		// Write the number of nodes to the file
		outFile << numNodes << endl;

		// Write edges to the file
		for (int i = 2; i <= numNodes; ++i) {
			int parentID = rand() % (i - 1) + 1; // Randomly select a parent ID from existing nodes
			outFile << parentID << " " << i << endl;
		}

		outFile.close();
		// cout << "Edges written to " << filename << endl;
	} else {
		cerr << "Unable to open file: " << filename << endl;
	}
}

__host__
void parseTree(const string& inputFile, int &n, int* &vertices, int* &parents, int* &children, int* &degree)
{
	fstream f;
	f.open(inputFile, ios::in);

	f >> n;

	vertices = new int[n+1];
	parents = new int[n+1];
	degree = new int[n+1];
	children = new int[(n-1)];
	vertices[0] = -1;
	parents[0] = -1;
	parents[1]=-1;
	vector<int> *g = new vector<int>[n + 1];

	int u,v;
	for(int i = 0; i < n-1; i ++)
	{
		f >> u >> v;
		g[u].push_back(v);
		parents[v] = u;
	}

	int ctr = 0;
	for(int i = 1; i <=n; i ++)
	{
		vertices[i] = ctr;
		for(auto &x: g[i])
		{
			children[ctr++] = x;
		}
		degree[i] = (int)g[i].size();
		if(degree[i]==0)
			vertices[i]=-1;
	}
}

__global__
void sixColoringTrees(int n, int* vertices, int* parents, int* colors, int* newColors)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if(id==0) // Do nothing on id = 0
		return;
	if(id>n) // Do nothing when id is more than n
		return;

	if(id==1) // No need to change color of root
		return;

	int myColor = colors[id];

	int bitIndex = __ffs(myColor ^ colors[parents[id]]);
	int myBit = (myColor & (1 << (bitIndex-1))) >> (bitIndex-1);
	newColors[id] = ((bitIndex << 1) ^ myBit);
}

__global__
void updateColors(int n, int* colors, int* newColors)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id==0) // Do nothing when id = 0
		return;
	if(id>n) // Do nothing when id is more than n
		return;
	colors[id] = newColors[id];
	// printf("blahhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh id:%d, NewColor: %d\n", id, newColors[id]);
}

__global__
void getParentColor(int n, int* parents, int* colors, int* newColors)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if(id==0) // Do nothing on id = 0
		return;
	if(id>n) // Do nothing when id is more than n
		return;
	if(id==1)
	{
		newColors[id] = 2; // Change color of root
		return;
	}

	newColors[id] = colors[parents[id]]; // Each node updates its color to the old color of its parents
	// printf("id: %d parent color: %d\n", id, colors[parents[id]]);
}

__global__
void badVertices(int n, int* vertices, int* parents, int* children, int* degree, int* colors, int* newColors, bool* badFlag)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id==0) // Do nothing when id = 0
		return;
	if(id>n) // Do nothing when id is more than n
		return;
	if(id==1)
	{
		badFlag[id] = 0;
		return;
	}

	int myColor = colors[id];
	if(myColor<=3)
	{
		badFlag[id] = 0;
		return;
	}

	int parentColor = colors[parents[id]];
	if(parentColor <= 3)
	{
		badFlag[id] = 1;
		return;
	}

	int childColor = 0;
	int no_of_children = degree[id];
	int start = vertices[id];
	for(int i=0;i<no_of_children;++i)
	{
		childColor = colors[children[start+i]];
		if(childColor <= 3)
		{
			badFlag[id] = 1;
			return;
		}
	}


	newColors[id] = newColors[id] - 3; // Making its color good, if all the neighbors are bad
	badFlag = 0;
	return;
}

// __global__
// void coloringGoodVertices(int n, int* colors, int* newColors, bool* badFlag)
// {
// 	int id = threadIdx.x + blockIdx.x * blockDim.x;
// 	if(id==0) // Do nothing when id = 0
// 		return;
// 	if(id>n) // Do nothing when id is more than n
// 		return;
// 	if(id==1)
// 		return;

// 	if(colors[id] > 3 && !badFlag[id])
// 	{
// 		newColors[id] = colors[id]-3;
// 		return;
// 	}
// 	if(badFlag[id])
// 		newColors[id] = 0;
// 	return;
// }

// Partially colors bad vertices whose parents are not bad.
__global__
void partialColoringBadVertices(int n, int* vertices, int* parents, int* children, int* degree, int* colors, int* newColors, bool* badFlag)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id==0) // Do nothing when id = 0
		return;
	if(id>n) // Do nothing when id is more than n
		return;
	if(id==1)
		return;

	if(badFlag[id])
	{
		int parentID = parents[id];
		if(!badFlag[parentID]) // Parent is not a bad vertex
		{
			// badFlag[id] = 0;
			int parentColor = colors[parentID];
			int childColor = 0;
			int no_of_children = degree[id];
			int start = vertices[id];
			for(int i=0;i<no_of_children;++i)
			{
				childColor = colors[children[start+i]];
				if(childColor>0 && childColor<=3)
					break;
			}

			if(parentColor!=1 && childColor!=1)
			{
				newColors[id] = 1;
				return;
			}
			if(parentColor!=2 && childColor!=2)
			{
				newColors[id] = 2;
				return;
			}
			newColors[id] = 3;
			return;
		}
		return;
	}
	return;
}

// Colors all the remaining bad vertices
__global__
void threeColoring(int n, int* vertices, int* parents, int* children, int* degree, int* colors, int* newColors, bool* badFlag)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id==0) // Do nothing when id = 0
		return;
	if(id>n) // Do nothing when id is more than n
		return;
	if(id==1)
		return;

	if(badFlag[id])
	{
		badFlag[id] = 0;
		int parentID = parents[id];
		int parentColor = colors[parentID];
		int childColor = 0;
		int no_of_children = degree[id];
		int start = vertices[id];
		for(int i=0;i<no_of_children;++i)
		{
			childColor = colors[children[start+i]];
			if(childColor>0 && childColor<=3)
				break;
		}

		if(parentColor!=1 && childColor!=1)
		{
			newColors[id] = 1;
			return;
		}
		if(parentColor!=2 && childColor!=2)
		{
			newColors[id] = 2;
			return;
		}
		newColors[id] = 3;
		return;
	}
}

__global__
void isValidColor(int n, int *vertices, int *parents, int *children, int *degree, int *colors, bool *validColor)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id==0) // Do nothing when id = 0
		return;
	if(id>n) // Do nothing when id is more than n
		return;

	int myColor = colors[id];
	
	int no_of_children=degree[id];
	int start = vertices[id];
	int childColor = 0;
	for(int i=0;i<no_of_children;++i)
	{
		childColor = colors[children[start+i]];
		if(myColor == childColor)
		{
			validColor[0] = false;
			return;
		}
	}

	if(id>1)
	{
		int parentColor = colors[parents[id]];
		if(myColor == parentColor)
		{
			validColor[0] = false;
			return;
		}

	}
}

int main(int argc, char **argv)
{
	srand(static_cast<unsigned>(time(nullptr)));

	string num = argv[1];
	int numNodes = stoi(argv[1], NULL, 10);

	if (numNodes <= 0) {
		cout << "Invalid number of nodes. Please enter a positive integer." << endl;
		return 1;
	}

	// Output the edges directly to a file
	string filename = "random_tree_edges_"+num+".txt";
	writeEdgesToFile(filename, numNodes);

	int n;
	int* vertices = NULL;
	int* parents = NULL;
	int* children = NULL;
	int* degree = NULL;
	parseTree(filename, n, vertices, parents, children, degree);

	int* colors = (int*)malloc((n+1)*sizeof(int));
	bool* badFlag = (bool*)malloc((n+1)*sizeof(bool));
	bool validColor = true;

	// Initialization of colors
	for(int i=1;i<=n;++i)
	{
		colors[i] = i;
		badFlag[i] = 0;
		// newColors[i] = i;
	}

	int L = ceil(log2(n+1));

	// Cuda Memory Allocation
	int *vertices_gpu, *parents_gpu, *children_gpu, *degree_gpu, *colors_gpu, *newColors_gpu;
	bool *badFlag_gpu;
	bool *validColor_gpu;
	// int* L_gpu, delta_gpu;
	if(cudaMalloc(&vertices_gpu, (n+1)*sizeof(int)) != cudaSuccess)
		cout << "Cannot allocate memory for vertices" << endl;
	if(cudaMalloc(&parents_gpu, (n+1)*sizeof(int)) != cudaSuccess)
		cout << "Cannot allocate memory for parents" << endl;
	if(cudaMalloc(&children_gpu, (n-1)*sizeof(int)) != cudaSuccess)
		cout << "Cannot allocate memory for vertices" << endl;
	if(cudaMalloc(&degree_gpu, (n+1)*sizeof(int)) != cudaSuccess)
		cout << "Cannot allocate memory for degree" << endl;
	if(cudaMalloc(&colors_gpu, (n+1)*sizeof(int)) != cudaSuccess)
		cout << "Cannot allocate memory for colors" << endl;
	if(cudaMalloc(&newColors_gpu, (n+1)*sizeof(int)) != cudaSuccess)
		cout << "Cannot allocate memory for colors" << endl;
	if(cudaMalloc(&badFlag_gpu, (n+1)*sizeof(bool)) != cudaSuccess)
		cout << "Cannot allocate memory for bad flags" << endl;
	if(cudaMalloc(&validColor_gpu, sizeof(bool)) != cudaSuccess)
		cout << "Cannot allocate memory for validColor flag" << endl;

	// cout << "CUDA Memory allocated" << endl;

	// Send data from host to device.
	if(cudaMemcpy(vertices_gpu, vertices, (n+1)*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
		cout << "Cannot copy vertices to device" << endl;
	if(cudaMemcpy(parents_gpu, parents, (n+1)*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
		cout << "Cannot copy parents to device" << endl;
	if(cudaMemcpy(children_gpu, children, (n-1)*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
		cout << "Cannot copy children to device" << endl;
	if(cudaMemcpy(degree_gpu, degree, (n+1)*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
		cout << "Cannot copy degree to device" << endl;
	if(cudaMemcpy(colors_gpu, colors, (n+1)*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
		cout << "Cannot copy colors to device" << endl;
	if(cudaMemcpy(newColors_gpu, colors, (n+1)*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
		cout << "Cannot copy colors to device" << endl;
	if(cudaMemcpy(badFlag_gpu, badFlag, (n+1)*sizeof(bool), cudaMemcpyHostToDevice) != cudaSuccess)
		cout << "Cannot copy colors to device" << endl;
	if(cudaMemcpy(validColor_gpu, &validColor, sizeof(bool), cudaMemcpyHostToDevice) != cudaSuccess)
		cout << "Cannot copy colors to device" << endl;

	// cout << "Data sent to device" << endl;

	// Set up CUDA event timers
	float elapsedTime = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Mark the beginning of the timed segment
	cudaEventRecord(start);

	int blockDim = 256;
	int gridDim = ceil(n/(256.0));

	int count = 0;
	while(L > (ceil(log2(L))+1))
	{
		// cout << "L: " << L << endl;
		sixColoringTrees<<<gridDim, blockDim>>>(n, vertices_gpu, parents_gpu, colors_gpu, newColors_gpu);
		updateColors<<<gridDim, blockDim>>>(n, colors_gpu, newColors_gpu);

		L = (ceil(log2(L))+1);
		count++;
	}

	// cout << "Six coloring done" << endl;

	getParentColor<<<gridDim, blockDim>>>(n, parents_gpu, colors_gpu, newColors_gpu);
	updateColors<<<gridDim, blockDim>>>(n, colors_gpu, newColors_gpu);

	// cout << "Updated color of node to color of parent node" << endl;

	badVertices<<<gridDim, blockDim>>>(n, vertices_gpu, parents_gpu, children_gpu, degree_gpu, colors_gpu, newColors_gpu, badFlag_gpu);
	updateColors<<<gridDim, blockDim>>>(n, colors_gpu, newColors_gpu);

	// cout << "Colored good vertices" << endl;

	partialColoringBadVertices<<<gridDim, blockDim>>>(n, vertices_gpu, parents_gpu, children_gpu, degree_gpu, colors_gpu, newColors_gpu, badFlag_gpu);
	updateColors<<<gridDim, blockDim>>>(n, colors_gpu, newColors_gpu);

	// cout << "partially colored bad vertices" << endl;

	threeColoring<<<gridDim, blockDim>>>(n, vertices_gpu, parents_gpu, children_gpu, degree_gpu, colors_gpu, newColors_gpu, badFlag_gpu);
	updateColors<<<gridDim, blockDim>>>(n, colors_gpu, newColors_gpu);


	// Verification
	isValidColor<<<gridDim, blockDim>>>(n, vertices_gpu, parents_gpu, children_gpu, degree_gpu, colors_gpu, validColor_gpu);


	// Send data from device to host.
	if(cudaMemcpy(&validColor, validColor_gpu, sizeof(bool), cudaMemcpyDeviceToHost) != cudaSuccess)
		cout << "Cannot copy colors from device to host" << endl;

	// Mark the end of the timed segment
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	// Calculate the duration between the start and stop markers (in milliseconds)
	cudaEventElapsedTime(&elapsedTime, start, stop);
	// Clean up event resources
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// cout << "Data copied from device to host" << endl;

	cout << "Time elapsed: " << elapsedTime << " ms" << '\n';
	cout << "Number of iterations: " << count << endl;
	cout << "isValidColoring: " << validColor << endl;

	// for(int i=1;i<=n;++i)
	// {
	// 	cout << "Color of " << i << ": " << colors[i] << endl;
	// }


	cudaFree(vertices_gpu);
	cudaFree(parents_gpu);
	cudaFree(children_gpu);
	cudaFree(degree_gpu);
	cudaFree(colors_gpu);
	cudaFree(newColors_gpu);
	cudaFree(badFlag_gpu);
	cudaFree(validColor_gpu);
	return 0;
}