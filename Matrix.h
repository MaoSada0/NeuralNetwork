#pragma once
#include <iostream>
class Matrix
{
	double** matrix; // матрица
	int row; // строки
	int col; // столбцы

public:
	void Init(int row, int col); // инициализация матрицы
	void Rand(); // заполнение матрицы случайными числами
	static void Multi(const Matrix& m, const double* b, int n, double* c); // умножение матрицы на столбец, m - матрица, b - столбец, n - размерность b, c - результат, который хотим получить
	static void Multi_T(const Matrix& m, const double* b, int n, double* c); // то же самое только транспонированная матрица
	static void Sum(double* a, const double* b, int n);
	double& operator ()(int i, int j);

	friend std::ostream& operator << (std::ostream& os, const Matrix& m);
	friend std::istream& operator >> (std::istream& is, Matrix& m);

};

