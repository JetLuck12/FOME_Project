import sys
import numpy as np
import scipy.linalg as la
import json
try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
    QLabel, QSlider, QPushButton, QSpinBox, QHBoxLayout, QDoubleSpinBox, QCheckBox, QFileDialog)
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QPixmap
    print("Using PyQt6")
except ImportError:
    try:
        from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
    QLabel, QSlider, QPushButton, QSpinBox, QHBoxLayout, QDoubleSpinBox, QCheckBox, QFileDialog)
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QPixmap
        print("Using PyQt5")
    except ImportError:
        raise ImportError("Neither PyQt5 nor PyQt6 is installed. Please install one of them.")
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class InteractiveGraphWindow(QMainWindow):
    def __init__(self, x_data, y_data, numstates=1, title="Интерактивный график"):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(800, 600)

        # Создаем центральный виджет и макет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Создаем фигуру и канвас
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Добавление панели инструментов
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

        # Построение графика
        ax = self.figure.add_subplot(111)


        # Проверяем размерность y_data
        if len(y_data.shape) == 1:  # Одномерный массив
            ax.plot(x_data, y_data, label="Потенциал")
        else:  # Двумерный массив
            for i in range(y_data.shape[1]):
                ax.plot(x_data, y_data[:, i], label=f"Уровень энергии {i + 1}")

        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        ax.grid(True)

        # Обновляем компоновку графиков
        self.figure.tight_layout()


class DeltaWellApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Delta Potential Wells/Grids")
        self.resize(1200, 800)

        # Основные параметры
        self.x_min, self.x_max = -10, 10
        self.N = 2000
        self.x = np.linspace(self.x_min, self.x_max, self.N)
        self.h = self.x[1] - self.x[0]
        self.num_wells = 1  # Начальное количество ям
        self.amplitudes = [4] * 5
        self.positions = [0] * 5
        self.antisymmetric = False  # Начальные граничные условия - симметричные
        self.V = np.zeros_like(self.x)
        self.eigenvalues, self.eigenvectors = None, None
        self.numstates = 3
        self.axes_list = []

        # UI компоненты
        main_layout = QVBoxLayout()
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(main_layout)

        # Панель управления
        control_layout = QVBoxLayout()
        graph_layout = QVBoxLayout()
        main_layout.addLayout(control_layout)
        main_layout.addLayout(graph_layout)

        config_layout = QHBoxLayout()
        save_button = QPushButton("Сохранить конфигурацию")
        save_button.clicked.connect(self.save_configuration)
        load_button = QPushButton("Загрузить конфигурацию")
        load_button.clicked.connect(self.load_configuration)
        config_layout.addWidget(save_button)
        config_layout.addWidget(load_button)
        control_layout.addLayout(config_layout)

        # Границы оси X
        x_limits_layout = QHBoxLayout()
        x_min_label = QLabel("x_min:")
        self.x_min_spin = QDoubleSpinBox()
        self.x_min_spin.setRange(-100, 100)
        self.x_min_spin.setValue(self.x_min)
        self.x_min_spin.valueChanged.connect(self.update_x_limits)

        x_max_label = QLabel("x_max:")
        self.x_max_spin = QDoubleSpinBox()
        self.x_max_spin.setRange(-100, 100)
        self.x_max_spin.setValue(self.x_max)
        self.x_max_spin.valueChanged.connect(self.update_x_limits)

        x_limits_layout.addWidget(x_min_label)
        x_limits_layout.addWidget(self.x_min_spin)
        x_limits_layout.addWidget(x_max_label)
        x_limits_layout.addWidget(self.x_max_spin)
        control_layout.addLayout(x_limits_layout)

        # Количество ям
        num_wells_layout = QHBoxLayout()
        num_wells_label = QLabel("Количество дельта ям:")
        num_states_label = QLabel("Количество решений:")
        self.num_wells_spin = QSpinBox()
        self.num_states_spin = QSpinBox()
        self.num_wells_spin.setRange(1, 5)
        self.num_states_spin.setRange(1, 5)
        self.num_wells_spin.setValue(self.num_wells)
        self.num_states_spin.setValue(self.numstates)
        self.num_wells_spin.valueChanged.connect(self.update_well_count)
        self.num_states_spin.valueChanged.connect(self.update_num_states)
        num_wells_layout.addWidget(num_wells_label)
        num_wells_layout.addWidget(self.num_wells_spin)
        num_wells_layout.addWidget(num_states_label)
        num_wells_layout.addWidget(self.num_states_spin)
        control_layout.addLayout(num_wells_layout)

        # Кнопка для равномерного распределения ям
        equal_spacing_layout = QHBoxLayout()
        spacing_label = QLabel("Расстояние между ямами:")
        self.spacing_spin = QDoubleSpinBox()
        self.spacing_spin.setRange(0.1, 50)
        self.spacing_spin.setValue(2)
        self.spacing_button = QPushButton("Выставить расстояние")
        self.spacing_button.clicked.connect(self.set_equal_spacing)
        equal_spacing_layout.addWidget(spacing_label)
        equal_spacing_layout.addWidget(self.spacing_spin)
        equal_spacing_layout.addWidget(self.spacing_button)
        control_layout.addLayout(equal_spacing_layout)

        # Кнопка для одинаковой амплитуды
        equal_amplitude_layout = QHBoxLayout()
        amplitude_label = QLabel("Амплитуда ям:")
        self.amplitude_spin = QDoubleSpinBox()
        self.amplitude_spin.setRange(-10, 10)
        self.amplitude_spin.setValue(4)
        self.amplitude_button = QPushButton("Выставить амплитуды")
        self.amplitude_button.clicked.connect(self.set_equal_amplitudes)
        equal_amplitude_layout.addWidget(amplitude_label)
        equal_amplitude_layout.addWidget(self.amplitude_spin)
        equal_amplitude_layout.addWidget(self.amplitude_button)
        control_layout.addLayout(equal_amplitude_layout)

        # Ползунки для амплитуд и позиций
        self.well_controls = []
        for i in range(5):
            well_layout = QHBoxLayout()
            amp_label = QLabel(f"Амплитуда ямы {i + 1}:")
            amp_slider = QDoubleSpinBox()
            amp_slider.setRange(-10, 10)
            amp_slider.setSingleStep(0.1)
            amp_slider.setValue(4)
            amp_slider.valueChanged.connect(self.update_amplitude)

            pos_label = QLabel(f"Позиция ямы {i + 1}:")
            pos_slider = QDoubleSpinBox()
            pos_slider.setRange(self.x_min, self.x_max)
            pos_slider.setSingleStep(0.1)
            pos_slider.setValue(0)
            pos_slider.valueChanged.connect(self.update_position)

            well_layout.addWidget(amp_label)
            well_layout.addWidget(amp_slider)
            well_layout.addWidget(pos_label)
            well_layout.addWidget(pos_slider)
            control_layout.addLayout(well_layout)

            self.well_controls.append((amp_slider, pos_slider))

        # Переключатель типа потенциала
        self.potential_selector = QCheckBox("Гармонический осциллятор (проверка)")
        self.potential_selector.stateChanged.connect(self.plot_graphs)
        control_layout.addWidget(self.potential_selector)

        # Переключатель граничных условий
        self.boundary_condition_checkbox = QCheckBox("Антисимметричные граничные условия")
        self.boundary_condition_checkbox.stateChanged.connect(self.toggle_boundary_conditions)
        control_layout.addWidget(self.boundary_condition_checkbox)

        # Кнопка "ОК"
        self.ok_button = QPushButton("ОК")
        self.ok_button.clicked.connect(self.plot_graphs)
        control_layout.addWidget(self.ok_button)



        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        graph_layout.addWidget(self.canvas)
        self.axes_list = []
        for i in range(1,4):
            ax = self.figure.add_subplot(3,1,i)
            self.axes_list.append(ax)  # Сохраняем ось

        self.filter_mode = None  # None означает, что фильтра нет (отображать все)

        # Начальная отрисовка
        self.plot_graphs()

    def save_configuration(self):
        """Сохраняет текущую конфигурацию в JSON-файл."""
        print("JIJA")
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить конфигурацию", "","JSON Files (*.json);;All Files (*)")
        if file_path:
            config = {
                "window_size": (self.width(), self.height()),
                "num_wells": self.num_wells,
                "amplitudes": self.amplitudes[:self.num_wells],
                "positions": self.positions[:self.num_wells],
                "x_min": self.x_min,
                "x_max": self.x_max,
                "num_states": self.numstates
            }
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(config, file, ensure_ascii=False, indent=4)
            print(f"Конфигурация сохранена в {file_path}")

    def load_configuration(self):
        """Загружает конфигурацию из JSON-файла и обновляет UI."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Загрузить конфигурацию", "","JSON Files (*.json);;All Files (*)")
        if file_path:
            with open(file_path, "r", encoding="utf-8") as file:
                config = json.load(file)
            print(f"Конфигурация загружена из {file_path}")

            # Применение конфигурации
            self.resize(*config["window_size"])
            self.num_wells = config["num_wells"]
            self.amplitudes[:self.num_wells] = config["amplitudes"]
            self.positions[:self.num_wells] = config["positions"]
            self.x_min = config["x_min"]
            self.x_max = config["x_max"]
            self.numstates = config["num_states"]

            # Обновление UI
            self.x_min_spin.setValue(self.x_min)
            self.x_max_spin.setValue(self.x_max)
            self.num_wells_spin.setValue(self.num_wells)
            self.num_states_spin.setValue(self.numstates)
            for i, (amp, pos) in enumerate(zip(self.amplitudes, self.positions)):
                self.well_controls[i][0].setValue(amp)
                self.well_controls[i][1].setValue(pos)

            # Перерисовка графиков
            self.plot_graphs()

    def on_click(self, event):
        if event.inaxes is not None:
            clicked_axis = event.inaxes

            print(self.axes_list[0])
            print(clicked_axis == self.axes_list[0])

            # Определяем, какой график был нажат
            for i, ax in enumerate(self.axes_list):
                if clicked_axis is ax:
                    print(f"Был произведён клик на графике {i + 1}")

                    x_data = self.x  # Данные оси X для графика
                    y_data = None
                    if i == 0:
                        y_data = self.V  # Потенциал
                    elif i == 1:
                        y_data = self.eigenvectors[:, :self.numstates]  # Первый волновой уровень
                    elif i == 2:
                        y_data = self.eigenvectors[:, :self.numstates] ** 2  # Квадрат первого уровня
                    # Открываем новое окно с интерактивным графиком
                    if i != 0:
                        self.graph_window = InteractiveGraphWindow(x_data, y_data,3, f"График {i + 1}")
                    else:
                        self.graph_window = InteractiveGraphWindow(x_data, y_data, 3, f"График {i}")

                    self.graph_window.show()
                    break

    def update_num_states(self):
        self.numstates = self.num_states_spin.value()

    def update_x_limits(self):
        self.x_min = self.x_min_spin.value()
        self.x_max = self.x_max_spin.value()
        self.x = np.linspace(self.x_min, self.x_max, self.N)
        self.h = self.x[1] - self.x[0]
        for _, pos_slider in self.well_controls:
            pos_slider.setRange(self.x_min, self.x_max)

    def set_equal_spacing(self):
        spacing = self.spacing_spin.value()
        start_position = -((self.num_wells - 1) * spacing) / 2
        for i in range(self.num_wells):
            self.positions[i] = start_position + i * spacing
            self.well_controls[i][1].setValue(self.positions[i])

    def set_equal_amplitudes(self):
        amplitude = self.amplitude_spin.value()
        for i in range(self.num_wells):
            self.amplitudes[i] = amplitude
            self.well_controls[i][0].setValue(amplitude)
        self.plot_graphs()

    def update_well_count(self):
        self.num_wells = self.num_wells_spin.value()
        for i, (amp_slider, pos_slider) in enumerate(self.well_controls):
            amp_slider.setEnabled(i < self.num_wells)
            pos_slider.setEnabled(i < self.num_wells)

    def update_amplitude(self):
        self.amplitudes = [slider.value() for slider, _ in self.well_controls]

    def update_position(self):
        self.positions = [slider.value() for _, slider in self.well_controls]

    def toggle_boundary_conditions(self):
        self.antisymmetric = self.boundary_condition_checkbox.isChecked()

    def plot_graphs(self):

        def is_symmetric(wavefunction):
            return np.allclose(wavefunction, wavefunction[::-1])

        def is_antisymmetric(wavefunction):
            return np.allclose(wavefunction, -wavefunction[::-1])


        # Формируем потенциал
        if self.potential_selector.isChecked():  # Если выбран гармонический осциллятор
            k = 1  # Коэффициент жесткости
            self.V = 0.5 * k * self.x ** 2
        else:
            self.V = np.zeros_like(self.x)
            for pos, amp in zip(self.positions[:self.num_wells], self.amplitudes[:self.num_wells]):
                idx = np.argmin(np.abs(self.x - pos))
                self.V[idx] += amp / self.h

        # Построим трехдиагональную матрицу
        T_coeff = -1 / (2 * self.h ** 2)
        H = np.zeros((self.N, self.N))
        np.fill_diagonal(H, -2 * T_coeff + self.V)
        np.fill_diagonal(H[1:], T_coeff)
        np.fill_diagonal(H[:, 1:], T_coeff)

        if self.antisymmetric:
            # Учет антисимметричных граничных условий
            H[0, 0] = H[-1, -1] = 1e6  # Высокий потенциал на краях
            H[0, 1] = H[-1, -2] = 0
            H[1, 0] = H[-2, -1] = 0

        # Найдем собственные значения и функции
        self.eigenvalues, self.eigenvectors = la.eigh(H)
        self.eigenvalues = self.eigenvalues[:self.numstates]
        self.eigenvectors = self.eigenvectors[:, :self.numstates]

        self.figure.clear()

        # График 1: Потенциал
        self.axes_list[0].set_title("Потенциал")
        self.axes_list[0].plot(self.x, self.V, label="Потенциал", color="black")
        self.axes_list[0].set_title("Потенциал")
        self.axes_list[0].set_xlabel("x")
        self.axes_list[0].set_ylabel("V(x)")
        self.axes_list[0].grid()
        self.axes_list[0].legend()
        self.figure.add_subplot(self.axes_list[0])

        # График 2: Волновые функции
        for i in range(self.numstates):
            psi = self.eigenvectors[:, i]
            psi /= np.sqrt(np.sum(psi ** 2) * self.h)
            self.axes_list[1].plot(self.x, psi + self.eigenvalues[i], label=f"ψ{i}, E={self.eigenvalues[i]:.3f}")
        self.axes_list[1].set_title("Волновые функции с энергией")
        self.axes_list[1].set_xlabel("x")
        self.axes_list[1].set_ylabel("ψ(x)")
        self.axes_list[1].grid()
        self.axes_list[1].legend()
        self.figure.add_subplot(self.axes_list[1])

        # График 3: Квадрат волновых функций
        self.axes_list[2].clear()
        for i in range(self.numstates):
            psi = self.eigenvectors[:, i]
            psi /= np.sqrt(np.sum(psi ** 2) * self.h)
            self.axes_list[2].plot(self.x, psi ** 2 + self.eigenvalues[i], label=f"|ψ{i}|², E={self.eigenvalues[i]:.3f}")
        self.axes_list[2].set_title("Квадрат волновых функций с энергией")
        self.axes_list[2].set_xlabel("x")
        self.axes_list[2].set_ylabel("|ψ(x)|²")
        self.axes_list[2].grid()
        self.axes_list[2].legend()
        self.figure.add_subplot(self.axes_list[2])

        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeltaWellApp()
    window.show()
    sys.exit(app.exec())
