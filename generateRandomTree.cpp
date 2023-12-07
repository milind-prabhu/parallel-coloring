#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <string>

using namespace std;

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

int main(int argc, char const *argv[]) {
	srand(static_cast<unsigned>(time(nullptr)));

	string n = argv[1];
	int numNodes = stoi(argv[1], NULL, 10);

	if (numNodes <= 0) {
		cout << "Invalid number of nodes. Please enter a positive integer." << endl;
		return 1;
	}

	// Output the edges directly to a file
	string filename = "random_tree_edges_"+n+".txt";
	writeEdgesToFile(filename, numNodes);

	return 0;
}
