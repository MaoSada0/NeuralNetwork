#include "NetWork.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <chrono>

struct data_info {
	double* pixels; // пиксели цифры
	int digit; // цифра 0-9
};

data_NetWork ReadDataNetWork(string path) {
	data_NetWork data{};

	ifstream fin;

	fin.open(path);

	if (!fin.is_open()) {
		cout << "Err reading file " << path << endl;
	}

	string tmp;
	int L;

	while (!fin.eof()) {
		fin >> tmp;
		if (tmp == "NetWork") {
			fin >> L;
			data.Layers = L;
			data.size = new int[L];

			for (int i = 0; i < L; i++) {
				fin >> data.size[i];
			}
		}
	}

	fin.close();

	return data;
}

data_info* ReadData(string path, const data_NetWork& data_NW, int& examples) {
	data_info* data;
	ifstream fin;
	fin.open(path);

	if (!fin.is_open()) {
		cout << "Err reading file " << path << endl;
	}
	else {
		if (path != "files/output.txt") {
			cout << path << " load... \n";
		}
	}

	string tmp;
	fin >> tmp;

	if (tmp == "Examples") {
		fin >> examples;
		data = new data_info[examples];

		for (int i = 0; i < examples; ++i) {
			data[i].pixels = new double[data_NW.size[0]];
		}

		for (int i = 0; i < examples; ++i) {
			fin >> data[i].digit;
			for (int j = 0; j < data_NW.size[0]; ++j) {
				fin >> data[i].pixels[j];
			}
		}
		fin.close();
		return data;
	}
	else {
		cout << "Error loading: " << path << endl;
		fin.close();
		return nullptr;
	}
}

void checkNum() {
	cout << "Enter name of file (name.png): ";
	string s;
	cin >> s;
	s = "numbers/" + s;

	cv::Mat image = cv::imread(s, cv::IMREAD_GRAYSCALE);

	while (image.empty()) {
		std::cerr << "Failed to load the image." << std::endl;
		cout << "Enter name of file (name.png): ";
		string s;
		cin >> s;
		s = "numbers/" + s;
		image = cv::imread(s, cv::IMREAD_GRAYSCALE);
	}

	std::ofstream outputFile("files/output.txt");

	if (!outputFile.is_open()) {
		std::cerr << "Failed to open the output file." << std::endl;
	}

	outputFile << "Examples 1" << std::endl;
	outputFile << "1" << std::endl;

	for (int i = 0; i < image.rows; ++i) {
		for (int j = 0; j < image.cols; ++j) {
			float brightness = 1 - (static_cast<float>(image.at<uchar>(i, j)) / 255.0f);
			outputFile << brightness << " ";
		}
		outputFile << std::endl;
	}

	outputFile.close();

}

int main() {
	NetWork NW{};
	data_NetWork NW_config;
	data_info* data;

	double ra = 0; // right answer
	double right = 0; // правильная цифра
	double predict = 0; // предсказанная цифра
	double maxra = 0; // макс колво правильных ответов за один заход
	int count = 0; // эпохи, заходы короче

	bool study = true;
	bool repeat = true;

	chrono::duration<double> time;

	NW_config = ReadDataNetWork("files/Settings.txt");
	NW.Init(NW_config);
	NW.PrintSettings();

	while (true) {
		cout << "Study? (1/0)" << endl;
		cin >> study;

		if (study) {
			int examples;
			data = ReadData("files/lib_MNIST_edit.txt", NW_config, examples);

			auto begin = chrono::steady_clock::now();

			while (ra / examples * 100 < 100) {
				ra = 0;
				auto t1 = chrono::steady_clock::now();
				for (int i = 0; i < examples; ++i) { // обучаем нейросеть идем по примерам в файле
					NW.SetInput(data[i].pixels); // пиксели примеров
					right = data[i].digit; // правильная цифра
					predict = NW.ForwardFeed(); // предсказываем цифру
					if (predict != right) { // обучаем если не совпадает
						NW.BackPropogation(right);
						NW.WeightsUpdater(0.15 * exp(-count / 20.));
					}
					else {
						ra++;
					}
				}
				auto t2 = chrono::steady_clock::now();
				time = t2 - t1;
				if (ra > maxra) {
					maxra = ra;
				}
				cout << "ra: " << ra / examples * 100 << "\t" << "maxra: " << maxra / examples * 100 << "\t" << "epoch: " << count << "\tTIME: " << time.count() << endl;
				count++;
				if (count == 25) { // тк до 100% никогда не дойдем выйдем после 25 обучений
					break;
				}
			}
			auto end = chrono::steady_clock::now();
			time = end - begin;
			cout << "TIME: " << time.count() / 60. << " min" << endl;
			NW.SaveWeights();
		}
		else {
			NW.ReadWeights();
		}

		cout << "Test? (1/0)\n";
		bool test_flag;
		cin >> test_flag;

		if (test_flag) {
			int ex_tests;
			data_info* data_test;
			data_test = ReadData("files/lib_10k.txt", NW_config, ex_tests);
			ra = 0;
			for (int i = 0; i < ex_tests; ++i) {
				NW.SetInput(data_test[i].pixels);
				predict = NW.ForwardFeed();
				right = data_test[i].digit;
				if (right == predict)
					ra++;
			}
			cout << "RA: " << ra / ex_tests * 100 << endl;
		}

		while (true) {
			checkNum();

			int ex_tests;
			data_info* data_test;
			data_test = ReadData("files/output.txt", NW_config, ex_tests);

			for (int i = 0; i < ex_tests; ++i) {
				NW.SetInput(data_test[i].pixels);
				predict = NW.ForwardFeed();
			}

			cout << "PREDICT: " << predict << endl;
		}
	}
	system("pause");
	return 0;
}