#pragma once
#include <iostream>
class Matrix
{
	double** matrix; // �������
	int row; // ������
	int col; // �������

public:
	void Init(int row, int col); // ������������� �������
	void Rand(); // ���������� ������� ���������� �������
	static void Multi(const Matrix& m, const double* b, int n, double* c); // ��������� ������� �� �������, m - �������, b - �������, n - ����������� b, c - ���������, ������� ����� ��������
	static void Multi_T(const Matrix& m, const double* b, int n, double* c); // �� �� ����� ������ ����������������� �������
	static void Sum(double* a, const double* b, int n);
	double& operator ()(int i, int j);

	friend std::ostream& operator << (std::ostream& os, const Matrix& m);
	friend std::istream& operator >> (std::istream& is, Matrix& m);

};

