#pragma once
#include <iostream>


class ActivateFunction // ReLu
{
public:
	void AF(double* value, int n); // вектор и его размер
	void AFDer(double* value, int n); // то же самое но производная
	double AFDer(double value);
};


