#include "NetWork.h"

void NetWork::Init(data_NetWork data) {
	srand(time(NULL));
	Layers = data.Layers;
	size = new int[Layers];
	for (int i = 0; i < Layers; i++) {
		size[i] = data.size[i];
	}

	weights = new Matrix[Layers - 1];
	bios = new double* [Layers - 1];

	for (int i = 0; i < Layers - 1; i++) {
		weights[i].Init(size[i + 1], size[i]);
		bios[i] = new double[size[i + 1]];
		weights[i].Rand();
		for (int j = 0; j < size[i + 1]; j++) {
			bios[i][j] = ((rand() % 50)) * 0.06 / (size[i] + 15);
		}
	}

	neurons_value = new double* [Layers];
	neurons_error = new double* [Layers];

	for (int i = 0; i < Layers; i++) {
		neurons_value[i] = new double[size[i]];
		neurons_error[i] = new double[size[i]];
	}

	neurons_bios_value = new double[Layers - 1];
	for (int i = 0; i < Layers - 1; i++) {
		neurons_bios_value[i] = 1;
	}
}

void NetWork::PrintSettings() {
	cout << "NeuralNetwork has " << Layers << " layers" << endl;
	cout << "Size: ";
	for (int i = 0; i < Layers; i++) {
		cout << i + 1 << ") " << size[i] << " ";
	}
	cout << endl;
	cout << endl;
}

void NetWork::SetInput(double* values) {
	for (int i = 0; i < size[0]; i++) {
		neurons_value[0][i] = values[i];
	}
}

double NetWork::ForwardFeed() {
	for (int k = 1; k < Layers; ++k) {
		Matrix::Multi(weights[k - 1], neurons_value[k - 1], size[k - 1], neurons_value[k]); 
		Matrix::Sum(neurons_value[k], bios[k - 1], size[k]);
		actFunc.AF(neurons_value[k], size[k]); 
	}
	int pred = SearchMaxIndex(neurons_value[Layers - 1]); 
	return pred; //ответ нейросети
}

int NetWork::SearchMaxIndex(double* value) { // находим индекс макс элемента в вектор-столбце
	double max = value[0];
	int prediction = 0;
	double tmp;
	for (int j = 1; j < size[Layers - 1]; j++) {
		tmp = value[j];
		if (tmp > max) {
			prediction = j;
			max = tmp;
		}
	}

	return prediction;
}

void NetWork::PrintValues(int L) {
	for (int j = 0; j < size[L]; j++) {
		cout << j << " " << neurons_value[L][j] << endl;
	}
}

void NetWork::BackPropogation(double expect) {
	for (int i = 0; i < size[Layers - 1]; i++) { // считаем дельту для выходных нейронов
		if (i != (int)expect) {
			neurons_error[Layers - 1][i] = -neurons_value[Layers - 1][i] * actFunc.AFDer(neurons_value[Layers - 1][i]);
		}
		else {
			neurons_error[Layers - 1][i] = (1.0 - neurons_value[Layers - 1][i]) * actFunc.AFDer(neurons_value[Layers - 1][i]);
		}
	}

	for (int k = Layers - 2; k > 0; k--) { // считаем дельту для скрытых нейронов
		Matrix::Multi_T(weights[k], neurons_error[k + 1], size[k + 1], neurons_error[k]);
		for (int j = 0; j < size[k]; j++) {
			neurons_error[k][j] *= actFunc.AFDer(neurons_value[k][j]);
		}
	}
}

void NetWork::WeightsUpdater(double lr) {
	for (int i = 0; i < Layers - 1; ++i) {
		for (int j = 0; j < size[i + 1]; ++j) {
			for (int k = 0; k < size[i]; ++k) {
				weights[i](j, k) += neurons_value[i][k] * neurons_error[i + 1][j] * lr;
			}
		}
	}
	for (int i = 0; i < Layers - 1; i++) {
		for (int k = 0; k < size[i + 1]; k++) {
			bios[i][k] += neurons_error[i + 1][k] * lr;
		}
	}
}

void NetWork::SaveWeights() {
	ofstream fout;
	fout.open("files/Weights.txt");
	if (!fout.is_open()) {
		cout << "Error reading the file";
		system("pause");
	}
	for (int i = 0; i < Layers - 1; ++i)
		fout << weights[i] << " ";

	for (int i = 0; i < Layers - 1; ++i) {
		for (int j = 0; j < size[i + 1]; ++j) {
			fout << bios[i][j] << " ";
		}
	}
	cout << "Weights saved \n";
	fout.close();
}

void NetWork::ReadWeights() {
	ifstream fin;
	fin.open("files/Weights.txt");
	if (!fin.is_open()) {
		cout << "Error reading the file";
		system("pause");
	}
	for (int i = 0; i < Layers - 1; ++i) {
		fin >> weights[i];
	}
	for (int i = 0; i < Layers - 1; ++i) {
		for (int j = 0; j < size[i + 1]; ++j) {
			fin >> bios[i][j];
		}
	}
	cout << "Weights readed \n";
	fin.close();
}