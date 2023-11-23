#pragma once
#include "ActivateFunction.h"
#include "Matrix.h"
#include <fstream>
using namespace std;

struct data_NetWork
{
	int L;
	int* size;
};

class NetWork
{
	int L; // кол-во слоев
	int* size; // кол-во нейронов на слоях
	ActivateFunction actFunc;
	Matrix* weights; // матрица весов
	double** bios; // веса смещения
	double** neurons_val; // значения нейронов
	double** neurons_err; // ошибка нейронов
	double* neurons_bios_val; // значения нейронов смещения

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

