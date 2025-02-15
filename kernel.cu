
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>
#include "cuda_runtime_api.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <malloc.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include <vector>

const int DEVICE = 0;
const int BUF_SIZE = 24;
const int WARTOWNIK = -1;

const char* DEFAULT_WORDS_FILE = "words5.txt";
const char* DEFAULT_SOLUTION_FILE = "solution.txt";

const char* E = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const int E_SIZE = 26;

using namespace std;
using namespace std::chrono;

cudaError_t warmUpWithCuda();
cudaError_t countDMatrix(int* D, char* S1, int l1, char* S2, int l2);
void loadWords(char** S1, char** S2, char* filePath);
vector<string> getEditInstructionsS1ToS2(char* S1, int l1, char* S2, int l2, int* D);
void saveEditInstructions(vector<string> instr, char* solutionPath);

// Funkcje pomocnicze (nieuzywane)
void printXMatrix(int* arr, int size, char* S1);
void printDMatrix(int* arr, char* S1, char* S2);
void printEditInstructions(vector<string> instr);
int generateRandStrings();
int generateS1string(char** S1);
int generateS2string(char** S2);
void saveDMatrix(int* D, int l1, int l2);
vector<string> getEditInstructionsS2ToS1(char* S1, int l1, char* S2, int l2, int* D);

__global__ void warmUpKernel() {
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float ia, ib;
	ia = ib = 0.0f;
	ib += ia + tid;
}

__global__ void setUpDMatrix(int* D, const int l1, const int l2)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j <= l1)
	{
		for (int i = 0; i <= l2; i++)
		{
			D[i * (l1 + 1) + j] = WARTOWNIK;
		}
	}
}

__global__ void countXMatrixKernel(int* X, const char* S1, const int l1, const char* E)
{
	int i = blockIdx.x * 1024 + threadIdx.x;
	int  xstart = i * (l1 + 1);
	X[xstart] = 0;
	for (int j = 1; j <= l1; j++)
	{
		X[xstart + j] = S1[j - 1] == E[i] ? j : X[xstart + j - 1];
	}
}

__global__ void countDMatrixKernel(int* D,
	const char* S1, const int l1,
	const char* S2, const int l2,
	const int* X)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int j = bid * blockDim.x + tid;
	bool isGlobalMemReading = j != 0 && j <= l1 && tid % 32 == 0;

	char S1_char;
	int Avar = 0, Bvar = 0, Cvar = 0, Dvar = j;
	int l, Xvar;
	char S2_i;

	// trzymanie słowa S2 w shared memory
	extern __shared__ char s_S2[];
	for (int k = tid; k < l2; k += blockDim.x)
		s_S2[k] = S2[k];

	// zapis potrzebnej litery i uzupelnienie pierwszego wiersza macierzy D
	if (0 <= j && j <= l1)
	{
		if (j != 0)
			S1_char = S1[j - 1];
		D[j] = Dvar;
	}

	for (int i = 1; i <= l2; i++)
	{
		// synchronizacja blokow
		while (isGlobalMemReading && D[(i - 1) * (l1 + 1) + j - 1] == WARTOWNIK) { /* busy waiting */ }

		__syncthreads();

		// pobieranie danych
		Avar = __shfl_up(Dvar, 1);

		if (j <= l1)
		{
			if (isGlobalMemReading)
			{
				Avar = D[(i - 1) * (l1 + 1) + j - 1];
			}

			Bvar = Dvar;
			S2_i = s_S2[i - 1];

			// obliczanie wedlug wzoru
			if (j == 0)
				Dvar = i;
			else if (S1_char == S2_i)
			{
				Dvar = Avar;
			}
			else
			{
				l = S2_i - 'A';
				Xvar = X[l * (l1 + 1) + j];

				if (Xvar == 0)
					Dvar = 1 + min(min(Avar, Bvar), i + j - 1);
				else
				{
					Cvar = D[(i - 1) * (l1 + 1) + Xvar - 1];
					Dvar = 1 + min(min(Avar, Bvar), Cvar + (j - 1 - Xvar));
				}
			}

			// zapis wyniku
			D[i * (l1 + 1) + j] = Dvar;
		}
	}
}

int main(int argc, char** argv)
{
	auto ts = high_resolution_clock::now();

	char* S1, * S2;
	int l1, l2;

#pragma region ProgramInnit
	printf("Usage: %s <words file name> <solution file name>\n", argv[0]);

	char* wordsPath;
	char* solutionsPath;
	if (argc < 2)
	{
		wordsPath = (char*)DEFAULT_WORDS_FILE;
		printf("Default path for words is used : %s\n", wordsPath);
	}
	else
		wordsPath = argv[1];

	if (argc < 3)
	{
		solutionsPath = (char*)DEFAULT_SOLUTION_FILE;
		printf("Default path for solution is used : %s\n", solutionsPath);
	}
	else
		solutionsPath = argv[2];

	loadWords(&S1, &S2, wordsPath);
	l1 = strlen(S1);
	l2 = strlen(S2);
#pragma endregion

	int* D = new int[(l1 + 1) * (l2 + 1)];
	cudaError_t cudaStatus;

	warmUpWithCuda();

	countDMatrix(D, S1, l1, S2, l2);

#pragma region ProgramEnd
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	vector<string> S1ToS2 = getEditInstructionsS1ToS2(S1, l1, S2, l2, D);

	saveEditInstructions(S1ToS2, solutionsPath);

	delete[] D;
	delete[] S1;
	delete[] S2;
#pragma endregion

	auto te = high_resolution_clock::now();
	cout << "The entire program runtime:    " << setw(7) << 0.001 * duration_cast<microseconds>(te - ts).count() << " nsec" << endl;

	return 0;
}

cudaError_t warmUpWithCuda()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, DEVICE);

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(DEVICE);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "warm up: cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		goto Error;
	}

	auto err = cudaGetLastError();
	if (err)
		printf("Last cuda error before warm up: %d -> %s\n", err, cudaGetErrorString(err));
	int t = prop.maxThreadsPerBlock;
	warmUpKernel << <3, t >> > ();

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	err = cudaGetLastError();
	if (err)
		printf("Last cuda error after warmup: %d -> %s\n", err, cudaGetErrorString(err));

Error:
	cudaFree(0);

	return cudaStatus;
}

cudaError_t countDMatrix(int* D, char* S1, int l1, char* S2, int l2)
{
	auto ts = high_resolution_clock::now();
	auto te = high_resolution_clock::now();

	char* dev_S1;
	char* dev_S2;
	char* dev_E;
	int* dev_X;
	int* dev_D;
	cudaError_t cudaStatus;

	int matrixXSize = (l1 + 1) * E_SIZE;
	int matrixDSize = (l1 + 1) * (l2 + 1);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, DEVICE);
	int block_size = prop.maxThreadsPerBlock;

	cudaStatus = cudaSetDevice(DEVICE);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	ts = high_resolution_clock::now();
#pragma region CudaMallocMemcpy
	cudaStatus = cudaMalloc((void**)&dev_X, matrixXSize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_D, matrixDSize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_S1, l1 * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_S2, l2 * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_E, E_SIZE * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_S1, S1, l1 * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_S2, S2, l2 * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_E, E, E_SIZE * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
#pragma endregion
	te = high_resolution_clock::now();
	cout << "Cuda Malloc and Memcpy:    " << setw(7) << 0.001 * duration_cast<microseconds>(te - ts).count() << " nsec" << endl;

	ts = high_resolution_clock::now();
	countXMatrixKernel << <1, E_SIZE >> > (dev_X, dev_S1, l1, dev_E);

	int blockCount = 1 + l1 / block_size;

	setUpDMatrix << <blockCount, block_size >> > (dev_D, l1, l2);

	countDMatrixKernel << <blockCount, block_size, l2 * sizeof(char) >> > (dev_D, dev_S1, l1, dev_S2, l2, dev_X);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
		goto Error;
	}
	te = high_resolution_clock::now();
	cout << "GPU counting D matrix:    " << setw(7) << 0.001 * duration_cast<microseconds>(te - ts).count() << " nsec" << endl;

	ts = high_resolution_clock::now();
	cudaStatus = cudaMemcpy(D, dev_D, matrixDSize * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyDeviceToHost failed!");
		goto Error;
	}

Error:
	cudaFree(dev_X);
	cudaFree(dev_D);
	cudaFree(dev_E);
	cudaFree(dev_S1);
	cudaFree(dev_S2);
	te = high_resolution_clock::now();
	cout << "Copying solution and freeing memory:  " << setw(7) << 0.001 * duration_cast<microseconds>(te - ts).count() << " nsec" << endl;

	return cudaStatus;
}

void loadWords(char** S1, char** S2, char* filePath)
{
	ifstream file(filePath);

	// String to store each line of the file.
	string line1;
	string line2;

	if (file.is_open()) {
		getline(file, line1);
		*S1 = new char[line1.size() + 1];
		strcpy(*S1, line1.c_str());

		getline(file, line2);
		*S2 = new char[line2.size() + 1];
		strcpy(*S2, line2.c_str());

		// Close the file stream once all lines have been
		// read.
		file.close();
	}
	else {
		// Print an error message to the standard error
		// stream if the file cannot be opened.
		cerr << "Unable to open file!" << endl;
	}

	int max_length = 10;
	cout << "Loaded strings:" << endl;
	cout << "First word(" << line1.size() << "): " << line1.substr(0, line1.length() < max_length ? line1.length() : max_length) << endl;
	cout << "Second word(" << line2.size() << "): " << line2.substr(0, line2.length() < max_length ? line2.length() : max_length) << endl;
}

vector<string> getEditInstructionsS1ToS2(char* S1, int l1, char* S2, int l2, int* D)
{
	int m = l2;
	int n = l1;
	int dist = D[m * (l1 + 1) + n];
	cout << "Distance: " << dist << "\n";

	vector<string> instr = {};
	int Ivar, Dvar, Rvar, minimum;
	char* buf = new char[BUF_SIZE];
	while (m > 0 || n > 0)
	{
		Ivar = m - 1 >= 0 ? D[(m - 1) * (l1 + 1) + n] : INT_MAX;
		Dvar = n - 1 >= 0 ? D[m * (l1 + 1) + n - 1] : INT_MAX;
		Rvar = m - 1 >= 0 && n - 1 >= 0 ? D[(m - 1) * (l1 + 1) + n - 1] : INT_MAX;
		minimum = std::min({ Ivar, Dvar, Rvar });

		if (minimum == Rvar)
		{
			if (Rvar < D[m * (l1 + 1) + n])
			{
				snprintf(buf, BUF_SIZE, "R, %d, %c\0", m - 1, S2[m - 1]);
				instr.push_back(string(buf, buf + strlen(buf)));
			}
			m--;
			n--;
		}
		else if (minimum == Dvar)
		{
			snprintf(buf, BUF_SIZE, "D, %d, %c\0", m, S1[n - 1]);
			instr.push_back(string(buf, buf + strlen(buf)));
			n--;
		}
		else
		{
			snprintf(buf, BUF_SIZE, "I, %d, %c\0", m - 1, S2[m - 1]);
			instr.push_back(string(buf, buf + strlen(buf)));
			m--;
		}
	}
	free(buf);

	return instr;
}

vector<string> getEditInstructionsS2ToS1(char* S1, int l1, char* S2, int l2, int* D)
{
	int m = l2;
	int n = l1;
	int dist = D[m * (l1 + 1) + n];
	cout << "Distance: " << dist << "\n";

	vector<string> instr = {};
	int Ivar, Dvar, Rvar, minimum;
	char* buf = new char[BUF_SIZE];
	while (m > 0 || n > 0)
	{
		Ivar = n - 1 >= 0 ? D[m * (l1 + 1) + n - 1] : INT_MAX;
		Dvar = m - 1 >= 0 ? D[(m - 1) * (l1 + 1) + n] : INT_MAX;
		Rvar = m - 1 >= 0 && n - 1 >= 0 ? D[(m - 1) * (l1 + 1) + n - 1] : INT_MAX;
		minimum = std::min({ Ivar, Dvar, Rvar });

		if (minimum == Rvar)
		{
			if (Rvar < D[m * (l1 + 1) + n])
			{
				snprintf(buf, BUF_SIZE, "R, %d, %c\0", n - 1, S1[n - 1]);
				instr.push_back(string(buf, buf + strlen(buf)));
			}
			m--;
			n--;
		}
		else if (minimum == Dvar)
		{
			snprintf(buf, BUF_SIZE, "D, %d, %c\0", n, S2[m - 1]);
			instr.push_back(string(buf, buf + strlen(buf)));
			m--;
		}
		else
		{
			snprintf(buf, BUF_SIZE, "I, %d, %c\0", n - 1, S1[n - 1]);
			instr.push_back(string(buf, buf + strlen(buf)));
			n--;
		}
	}
	free(buf);

	return instr;
}

void saveEditInstructions(vector<string> instr, char* filePath)
{
	ofstream outfile;

	outfile.open(filePath, fstream::out);

	for (int i = instr.size() - 1; i >= 0; i--) {
		outfile << instr[i] << "\n";
	}

	outfile.close();
}

void printXMatrix(int* arr, int size, char* S1)
{
	int l1 = strlen(S1);

	cout << "X matrix:" << endl;

	cout << "  e ";
	for (int i = 0; i < l1; i++)
	{
		cout << S1[i] << ' ';
	}

	for (int i = 0; i < size; i++)
	{
		if (i % (l1 + 1) == 0)
			cout << endl << E[i / (l1 + 1)] << " ";

		cout << arr[i] << ' ';
	}
	cout << endl << endl;
}

void printDMatrix(int* arr, char* S1, char* S2)
{
	int l1 = strlen(S1);
	int l2 = strlen(S2);

	cout << "D matrix:" << endl;

	cout << "  e ";
	for (int i = 0; i < l1; i++)
	{
		cout << S1[i] << ' ';
	}

	for (int i = 0; i <= l2; i++)
	{
		if (i == 0)
			cout << endl << "e ";
		else
			cout << endl << S2[i - 1] << " ";

		for (int j = 0; j <= l1; j++)
			cout << arr[i * (l1 + 1) + j] << ' ';
	}
	cout << endl << endl;
}

void printEditInstructions(vector<string> instr)
{
	for (int i = instr.size() - 1; i >= 0; i--) {
		cout << instr[i] << "\n";
	}
}

int generateRandStrings()
{
	srand(time(nullptr));

	ofstream outfile("wordsG.txt");
	int size = 5000 + rand() % 2000;
	for (int i = 0; i < size; i++)
		outfile << E[rand() % E_SIZE];
	outfile << "\n";

	size = 5000 + rand() % 2000;
	for (int i = 0; i < size; i++)
		outfile << E[rand() % E_SIZE];
	outfile << "\n";

	outfile.close();
	return size;
}

int generateS1string(char** S1)
{
	int size = 2047;
	*S1 = new char[size];
	(*S1)[size] = '\0';
	for (int i = 0; i < size; i++)
	{
		(*S1)[i] = 'A';

		if (i == 800)
			(*S1)[i] = 'C';
	}
	return size;
}

int generateS2string(char** S2)
{
	int size = 2047;
	*S2 = new char[size];
	(*S2)[size] = '\0';
	for (int i = 0; i < size; i++)
	{
		(*S2)[i] = 'A';

		if (i == 1500)
			(*S2)[i] = 'B';
	}
	return size;
}

void saveDMatrix(int* D, int l1, int l2)
{
	ofstream outfile("DMatrix.txt");
	int size = (l1 + 1) * (l2 + 1);

	for (int i = 0; i < size; i++)
	{
		outfile << D[i] << " ";
		if ((i != 0) && ((i + 1) % (l1 + 1) == 0))
			outfile << '\n';
	}

	outfile.close();
}

