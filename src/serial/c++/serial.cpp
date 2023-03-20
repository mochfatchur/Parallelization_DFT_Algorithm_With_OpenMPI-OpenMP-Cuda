#include <complex>
#include <iostream>
#include <vector>

using namespace std::complex_literals;
using namespace std;

constexpr complex<double> pi() {
    return std::atan(1) * 4;
}

class Matrix {
    private:
        int n;
        vector<vector<double>> *data_ptr;
        vector<vector<double>> &data;
    public:
        Matrix() : data_ptr(new vector<vector<double>>()), data(*data_ptr) {
            this->data.resize(10, vector<double>(10, 0));
        }
        ~Matrix() {
            delete data_ptr;
        }

        void readMatrix() {
            cin >> this->n;
            this->data.resize(this->n, vector<double>(this->n, 0.0));
            for (int i = 0; i < this->n; i++)
                for (int j = 0; j < this->n; j++)
                    cin >> this->data[i][j];
        }
        int size() {
            return this->n;
        }
        complex<double> dftElement(int k, int l) {
            complex<double> element = 0;
            for (int m = 0; m < this->n; m++) {
                for (int n = 0; n < this->n; n++) {
                    complex<double> sample = (k*m / (double) this->n) + (l*n / (double) this->n);
                    complex<double> exponent = exp(-2.0i*pi() * sample);
                    element += this->data[m][n] * exponent;
                }
            }
            return element / (complex<double>)(this->n * this->n);
        }
};



int main(void) {
    Matrix *source = new Matrix();
    source->readMatrix();

    complex<double> sum = 0.0;
    for (int i = 0; i < source->size(); i++) {
        for (int j = 0; j < source->size(); j++) {
            complex<double> el = source->dftElement(i, j);
            sum                += el;
        }
    }
    cout << sum / (double) source->size();
    return 0;
}