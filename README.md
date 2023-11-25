# NeuralNetwork
Приветствую всех! Здесь я опишу структуру нейронной сети.

!!! ДЛЯ РАБОТЫ НУЖНА БИБЛИОТЕКА OPENCV !!!

## Структура файлов


**Папка files** - в ней хранятся txt файлы, нужные для проекта:
- output.txt - в него записывается матрица весов png картинки 28x28, которую дает пользователь
- Settings.txt - это файл конфига, в нем записано количество слоев, а также количество нейронов на этих слоях
- Weights.txt - это файл весов, которые получаются после обучения нейронной сети.
  
  Следующих файлов нет, в связи c их большим весом (>100мб) - это [данные для обучения](http://yann.lecun.com/exdb/mnist/)
- lib_MNIST_edit.txt - 60000 цифр для обучения нейронной сети
- lib_10k.txt - 10000 цифр для проверки работоспособности нейронной сети.

**Папка numbers** - сюда сохраняются пользовательские 28x28 png изображения, для того чтобы их угадывала нейросеть

**Остались только файлы кодов**

## Код

### Class Matrix
> Это класс для действий с матрицами

**Переменнные:**
- ````double** matrix```` - двумерный массив, то есть матрицы
- ````int row```` - строки
- ````int col```` - столбцы
  
**Методы:**
- ````Init(int row, int col)```` - инициализируем матрицу (выделяем память)
- ````Rand()```` - заполнение матрицы случайными числами
- ````Multi(const Matrix& m, const double* b, int n, double* c)```` - умножение матрицы на вектор-столбец
- ````Multi_T(const Matrix& m, const double* b, int n, double* c)```` - умножение транспонированной матрицы на вектор-столбец
- ````Sum(double* a, const double* b, int n)```` - суммируем вектор-столбцы
  
### Class ActivateFunction
> Это класс активационных функций (коротко АФ). Есть множество функций, я выбирал из ReLU, thx, sigmoid. В итоге я выбрал ReLU, тк на тестах она показала себя лучше всего. Также пришлось немного изменить функцию, чтобы ее область значений была (0;1)

 **Методы:**
- ````use(double* value, int n)```` - используем АФ для вектор-столбца
- ````useDer(double* value, int n)```` - используем производную АФ для вектор-столбца
- ````useDer(double value)```` - используем производную АФ для значения

### Class NetWork
> Это класс сети

**Структура data_NetWork**
- ````int Layers```` - количество слоев
- ````int* size```` - массив с количеством нейронов на каждом из слоев

**Переменнные:**
- ````int Layers```` - количество слоев
- ````int* size```` - массив с количеством нейронов на каждом из слоев
- ````ActivateFunction actFunc```` - АФ
- ````Matrix* weights```` - матрица весов
- ````double** bios```` - веса смещения
- ````double** neurons_value```` - значения нейронов
- ````double** neurons_error```` - ошибки нейронов
- ````double* neurons_bios_value```` - значения нейронов смещения
  
**Методы:**
- ````Init(data_NetWork data)```` - инициализируем нейросеть 
- ````PrintSettings()```` - выводим settings
- ````SetInput(double* values)```` - подаем на вход нейросети данные
- ````ForwardFeed()```` - функция прямого распростронения
- ````SearchMaxIndex(double* value)```` - ищем индекс max элемента вектора значений
- ````PrintValues```` - выводим значения на экран
- ````BackPropogation(double expect)````- функция обратного распростронения
- ````WeightsUpdater(double lr)````- обновление весов
- ````SaveWeights()````- сохраняем веса в файл
- ````ReadWeights()````- читаем веса из файла

### Neural (main)
> main файл, в нем реализуем работу с нейросетью

**Структура data_info**
- ````double* pixels```` - пиксели цифры
- ````int digit```` - цифра 0-9
  
**Методы:**
- ````ReadDataNetWork(string path)````- читаем settings
- ````ReadData(string path, const data_NetWork& data_NW, int& examples)```` - считываем данные цифр для нейронки
- ````void checkNum()```` - угадываем пользовательскую цифру
- ````main()```` - main


### Как работает данная нейросеть?

Начнем с того, что нейросеть нужно обучить. В начале мы получаем на входной слой 784 нейрона (28 x 28 пикселей). Заполняем матрицу весов и биасов случайными значениями. Запускаем функцию прямого распростроения и получаем выходные нейроны, там где max значение - там ответ. Ответ сошелся? Нет - запускаем функцию обратного распространения и корректируем веса, да - идем дальше. И так проходимя по всему обучающему файлу. Так как за один проход (далее - эпоха) нейросеть может плохо натренироваться, принято увеличить количество эпох. Все, после нейросеть обучена. Для того, чтобы она угадывала пользовательские цифры, я использовал библиотеку openCV. С помощью ее я делаю из 28x28 пиксельного изображения матрицу весов 28x28, далее просто подаю эту матрицу нейросети, она запускает функцию прямого распространения и выдает ответ.
