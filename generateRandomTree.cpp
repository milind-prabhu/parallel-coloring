#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>

using namespace std;

// Node structure for the tree
struct Node {
	int id;
	vector<Node*> children;
};

// Function to generate a random tree with a specified number of nodes
Node* generateRandomTree(int numNodes) {
	if (numNodes <= 0) {
		return nullptr;
	}

	Node* root = new Node{1}; // Root node ID is 1

	vector<Node*> nodes;  // Store nodes for easy access

	nodes.push_back(root);

	for (int i = 2; i <= numNodes; ++i) {
		int parentID = rand() % (i - 1) + 1; // Randomly select a parent ID from existing nodes
		Node* parent = nodes[parentID - 1]; // Access the correct parent by index

		Node* newNode = new Node{i}; // Child node ID is i
		parent->children.push_back(newNode);
		nodes.push_back(newNode);
	}

	return root;
}

// Function to get edges from the tree
void getEdges(const Node* root, vector<pair<int, int>>& edges) {
	if (!root) return;

	for (Node* child : root->children) {
		edges.emplace_back(root->id, child->id);
		getEdges(child, edges);
	}
}

// Function to write edges to a file
void writeEdgesToFile(const vector<pair<int, int>>& edges, const string& filename, int numNodes) {
	ofstream outFile(filename);

	if (outFile.is_open()) {
		// Write the number of nodes to the file
		outFile << numNodes << endl;

		// Write edges to the file
		for (const auto& edge : edges) {
			outFile << edge.first << " " << edge.second << endl;
		}

		outFile.close();
		cout << "Edges written to " << filename << endl;
	} else {
		cerr << "Unable to open file: " << filename << endl;
	}
}

int main() {
	srand(static_cast<unsigned>(time(nullptr)));

	int numNodes;
	cout << "Enter the number of nodes in the tree: ";
	cin >> numNodes;

	if (numNodes <= 0) {
		cout << "Invalid number of nodes. Please enter a positive integer." << endl;
		return 1;
	}

	Node* randomTree = generateRandomTree(numNodes);

	// Get edges from the randomly generated tree
	vector<pair<int, int>> edges;
	getEdges(randomTree, edges);

	// Sort edges based on the first node ID
	sort(edges.begin(), edges.end());

	// Output the edges of the randomly generated tree to a file
	string filename = "random_tree_edges.txt";
	writeEdgesToFile(edges, filename, numNodes);

	// TODO: Clean up the allocated memory for the tree nodes

	return 0;
}
