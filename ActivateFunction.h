#pragma once
#include <iostream>


class ActivateFunction // ReLu
{
public:
	void use(double* value, int n); // ������ � ��� ������
	void useDer(double* value, int n); // �� �� ����� �� �����������
	double useDer(double value);
};


