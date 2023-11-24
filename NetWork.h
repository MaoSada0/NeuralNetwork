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
	int Layers; // кол-во слоев
	int* size; // кол-во нейронов на слоях
	ActivateFunction actFunc;
	Matrix* weights; // матрица весов
	double** bios; // веса смещения
	double** neurons_value; // значения нейронов
	double** neurons_error; // ошибка нейронов
	double* neurons_bios_value; // значения нейронов смещения

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

