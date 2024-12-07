import sys
import numpy as np
import scipy.linalg as la
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QLabel, QSlider, QPushButton, QSpinBox, QHBoxLayout, QDoubleSpinBox, QCheckBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class DeltaWellApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Delta Potential Wells")
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

        # UI компоненты
        main_layout = QVBoxLayout()
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(main_layout)

        # Панель управления
        control_layout = QVBoxLayout()
        main_layout.addLayout(control_layout)

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
        self.num_wells_spin = QSpinBox()
        self.num_wells_spin.setRange(1, 5)
        self.num_wells_spin.setValue(self.num_wells)
        self.num_wells_spin.valueChanged.connect(self.update_well_count)
        num_wells_layout.addWidget(num_wells_label)
        num_wells_layout.addWidget(self.num_wells_spin)
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

        # Графики
        self.figures = [Figure() for _ in range(3)]
        self.canvases = [FigureCanvas(fig) for fig in self.figures]
        for canvas in self.canvases:
            main_layout.addWidget(canvas)

        self.filter_mode = None  # None означает, что фильтра нет (отображать все)

        # Начальная отрисовка
        self.plot_graphs()

    def update_x_limits(self):
        self.x_min = self.x_min_spin.value()
        self.x_max = self.x_max_spin.value()
        self.x = np.linspace(self.x_min, self.x_max, self.N)
        self.h = self.x[1] - self.x[0]
        for _, pos_slider in self.well_controls:
            pos_slider.setRange(self.x_min, self.x_max)
        self.plot_graphs()

    def set_equal_spacing(self):
        spacing = self.spacing_spin.value()
        start_position = -((self.num_wells - 1) * spacing) / 2
        for i in range(self.num_wells):
            self.positions[i] = start_position + i * spacing
            self.well_controls[i][1].setValue(self.positions[i])
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
            V = 0.5 * k * self.x ** 2
        else:
            V = np.zeros_like(self.x)
            for pos, amp in zip(self.positions[:self.num_wells], self.amplitudes[:self.num_wells]):
                idx = np.argmin(np.abs(self.x - pos))
                V[idx] += amp / self.h

        # Построим трехдиагональную матрицу
        T_coeff = -1 / (2 * self.h ** 2)
        H = np.zeros((self.N, self.N))
        np.fill_diagonal(H, -2 * T_coeff + V)
        np.fill_diagonal(H[1:], T_coeff)
        np.fill_diagonal(H[:, 1:], T_coeff)

        if self.antisymmetric:
            # Учет антисимметричных граничных условий
            H[0, 0] = H[-1, -1] = 1e6  # Высокий потенциал на краях
            H[0, 1] = H[-1, -2] = 0
            H[1, 0] = H[-2, -1] = 0

        # Найдем собственные значения и функции
        num_states = 3
        eigenvalues, eigenvectors = la.eigh(H)
        eigenvalues = eigenvalues[:num_states]
        eigenvectors = eigenvectors[:, :num_states]

        # График 1: Потенциал
        ax1 = self.figures[0].clear()
        ax1 = self.figures[0].add_subplot(111)
        ax1.plot(self.x, V, label="Потенциал", color="black")
        ax1.set_title("Потенциал")
        ax1.set_xlabel("x")
        ax1.set_ylabel("V(x)")
        ax1.grid()
        ax1.legend()

        # График 2: Волновые функции
        ax2 = self.figures[1].clear()
        ax2 = self.figures[1].add_subplot(111)
        for i in range(num_states):
            psi = eigenvectors[:, i]
            psi /= np.sqrt(np.sum(psi ** 2) * self.h)
            ax2.plot(self.x, psi + eigenvalues[i], label=f"ψ{i}, E={eigenvalues[i]:.3f}")
        ax2.set_title("Волновые функции с энергией")
        ax2.set_xlabel("x")
        ax2.set_ylabel("ψ(x)")
        ax2.grid()
        ax2.legend()

        # График 3: Квадрат волновых функций
        ax3 = self.figures[2].clear()
        ax3 = self.figures[2].add_subplot(111)
        for i in range(num_states):
            psi = eigenvectors[:, i]
            psi /= np.sqrt(np.sum(psi ** 2) * self.h)
            ax3.plot(self.x, psi ** 2 + eigenvalues[i], label=f"|ψ{i}|², E={eigenvalues[i]:.3f}")
        ax3.set_title("Квадрат волновых функций с энергией")
        ax3.set_xlabel("x")
        ax3.set_ylabel("|ψ(x)|²")
        ax3.grid()
        ax3.legend()

        # Обновление всех графиков
        for canvas in self.canvases:
            canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeltaWellApp()
    window.show()
    sys.exit(app.exec_())
