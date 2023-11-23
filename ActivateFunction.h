#pragma once
#include <iostream>


class ActivateFunction // ReLu
{
public:
	void use(double* value, int n); // вектор и его размер
	void useDer(double* value, int n); // то же самое но производная
	double useDer(double value);
};


