#include "ActivateFunction.h"


void ActivateFunction::AF(double* value, int n) {
	for (int i = 0; i < n; i++) {
		if (value[i] < 0) {
			value[i] = value[i] * 0.01;
		}
		else if (value[i] > 1) {
			value[i] = 1. + 0.01 * (value[i] - 1.);
		}
	}

}

double ActivateFunction::AFDer(double value) {
	if (value < 0 || value > 1) {
		value = 0.01;
	}
	return value;
}

void ActivateFunction::AFDer(double* value, int n) {
	for (int i = 0; i < n; i++) {
		if (value[i] < 0 || value[i] > 1) {
			value[i] = 0.01;
		}
		else {
			value[i] = 1;
		}
	}
}