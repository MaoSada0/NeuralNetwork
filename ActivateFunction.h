#pragma once
#include <iostream>


class ActivateFunction // ReLu
{
public:
	void AF(double* value, int n); // ������ � ��� ������
	void AFDer(double* value, int n); // �� �� ����� �� �����������
	double AFDer(double value);
};


