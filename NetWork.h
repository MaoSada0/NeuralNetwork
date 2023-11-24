#pragma once
#include "ActivateFunction.h"
#include "Matrix.h"
#include <fstream>
using namespace std;

struct data_NetWork
{
	int Layers;
	int* size;
};

class NetWork
{
	int Layers; // ���-�� �����
	int* size; // ���-�� �������� �� �����
	ActivateFunction actFunc;
	Matrix* weights; // ������� �����
	double** bios; // ���� ��������
	double** neurons_value; // �������� ��������
	double** neurons_error; // ������ ��������
	double* neurons_bios_value; // �������� �������� ��������

public:
	void Init(data_NetWork data);
	void PrintConfig();
	void SetInput(double* values);

	double ForwardFeed();
	int SearchMaxIndex(double* value);
	void PrintValues(int L);

	void BackPropogation(double expect);
	void WeightsUpdater(double lr);

	void SaveWeights();
	void ReadWeights();
};

