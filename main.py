import sys
import os
import pandas as pd
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QVBoxLayout,
                             QHBoxLayout, QWidget, QPushButton, QComboBox,
                             QLabel, QFileDialog, QTextEdit, QTableWidget,
                             QTableWidgetItem, QMessageBox, QScrollArea)
from PyQt5.QtCore import Qt
import numpy as np
from datetime import datetime


class DataVisualizationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.df = None
        self.db_connection = None
        self.current_table = None
        self.log_actions = []
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Анализатор данных из Kaggle')
        self.setGeometry(100, 100, 1400, 900)

        # Создание вкладок
        self.tabs = QTabWidget()

        # Вкладка 1: Статистика по данным
        self.tab1 = QWidget()
        self.setup_tab1()
        self.tabs.addTab(self.tab1, "📊 Статистика")

        # Вкладка 2: Графики корреляции
        self.tab2 = QWidget()
        self.setup_tab2()
        self.tabs.addTab(self.tab2, "📈 Корреляции")

        # Вкладка 3: Тепловая карта
        self.tab3 = QWidget()
        self.setup_tab3()
        self.tabs.addTab(self.tab3, "🔥 Тепловая карта")

        # Вкладка 4: Линейные графики
        self.tab4 = QWidget()
        self.setup_tab4()
        self.tabs.addTab(self.tab4, "📉 Линейные графики")

        # Вкладка 5: Лог действий
        self.tab5 = QWidget()
        self.setup_tab5()
        self.tabs.addTab(self.tab5, "📝 Лог действий")

        self.setCentralWidget(self.tabs)
        self.add_log("Приложение запущено. Загрузите ваш датасет!")

    def setup_tab1(self):
        layout = QVBoxLayout()

        # Панель загрузки данных
        load_layout = QHBoxLayout()
        self.load_btn = QPushButton('📁 Загрузить CSV (ваш датасет)')
        self.load_btn.clicked.connect(self.load_csv)
        self.load_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        load_layout.addWidget(self.load_btn)

        self.load_db_btn = QPushButton('🗃️ Загрузить из БД')
        self.load_db_btn.clicked.connect(self.load_from_db)
        load_layout.addWidget(self.load_db_btn)

        self.save_db_btn = QPushButton('💾 Сохранить в БД')
        self.save_db_btn.clicked.connect(self.save_to_db)
        load_layout.addWidget(self.save_db_btn)

        self.table_combo = QComboBox()
        self.table_combo.currentTextChanged.connect(self.load_table_data)
        load_layout.addWidget(QLabel('Таблица:'))
        load_layout.addWidget(self.table_combo)

        layout.addLayout(load_layout)

        # Информация о данных
        self.info_label = QLabel('Данные не загружены')
        self.info_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; }")
        layout.addWidget(self.info_label)

        # Отображение статистики
        layout.addWidget(QLabel('<b>Базовая статистика:</b>'))
        self.stats_table = QTableWidget()
        layout.addWidget(self.stats_table)

        # Отображение данных
        layout.addWidget(QLabel('<b>Первые 100 строк данных:</b>'))
        self.data_table = QTableWidget()
        layout.addWidget(self.data_table)

        self.tab1.setLayout(layout)

    def setup_tab2(self):
        layout = QVBoxLayout()

        controls_layout = QHBoxLayout()
        self.corr_plot_btn = QPushButton('📊 Построить графики корреляции')
        self.corr_plot_btn.clicked.connect(self.plot_correlation)
        self.corr_plot_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; }")
        controls_layout.addWidget(self.corr_plot_btn)

        layout.addLayout(controls_layout)

        # Область для графиков
        self.corr_scroll = QScrollArea()
        self.corr_widget = QWidget()
        self.corr_layout = QVBoxLayout(self.corr_widget)
        self.corr_scroll.setWidget(self.corr_widget)
        self.corr_scroll.setWidgetResizable(True)
        layout.addWidget(self.corr_scroll)

        self.tab2.setLayout(layout)

    def setup_tab3(self):
        layout = QVBoxLayout()

        controls_layout = QHBoxLayout()
        self.heatmap_btn = QPushButton('🔥 Построить тепловую карту')
        self.heatmap_btn.clicked.connect(self.plot_heatmap)
        self.heatmap_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; }")
        controls_layout.addWidget(self.heatmap_btn)

        layout.addLayout(controls_layout)

        # Область для тепловой карты
        self.heatmap_figure = Figure(figsize=(10, 8))
        self.heatmap_canvas = FigureCanvas(self.heatmap_figure)
        layout.addWidget(self.heatmap_canvas)

        self.tab3.setLayout(layout)

    def setup_tab4(self):
        layout = QVBoxLayout()

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel('Выберите столбец для графика:'))
        self.column_combo = QComboBox()
        controls_layout.addWidget(self.column_combo)

        self.line_plot_btn = QPushButton('📈 Построить линейный график')
        self.line_plot_btn.clicked.connect(self.plot_line_chart)
        self.line_plot_btn.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; }")
        controls_layout.addWidget(self.line_plot_btn)

        layout.addLayout(controls_layout)

        # Область для линейного графика
        self.line_figure = Figure(figsize=(10, 6))
        self.line_canvas = FigureCanvas(self.line_figure)
        layout.addWidget(self.line_canvas)

        self.tab4.setLayout(layout)

    def setup_tab5(self):
        layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        clear_btn = QPushButton('🧹 Очистить лог')
        clear_btn.clicked.connect(self.clear_log)
        layout.addWidget(clear_btn)

        self.tab5.setLayout(layout)

    def add_log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_actions.append(log_entry)
        self.log_text.append(log_entry)

    def clear_log(self):
        self.log_text.clear()
        self.log_actions = []
        self.add_log("Лог очищен")

    def load_csv(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                'Выберите ваш CSV файл из Kaggle',
                '',
                'CSV Files (*.csv);;All Files (*)'
            )
            if file_path:
                self.df = pd.read_csv(file_path)
                self.update_data_display()
                file_name = os.path.basename(file_path)
                self.add_log(f"✅ Загружен CSV файл: {file_name}")
                self.add_log(f"📏 Размер данных: {self.df.shape[0]} строк, {self.df.shape[1]} столбцов")
                QMessageBox.information(self, 'Успех',
                                        f'Данные загружены!\nСтрок: {self.df.shape[0]}\nСтолбцов: {self.df.shape[1]}')
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', f'Ошибка загрузки CSV: {str(e)}')
            self.add_log(f"❌ Ошибка загрузки: {str(e)}")

    def load_from_db(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, 'Выбрать базу данных', '', 'SQLite Files (*.db *.sqlite)')
            if file_path:
                self.db_connection = sqlite3.connect(file_path)
                self.update_table_list()
                self.add_log(f"🗃️ Подключена база данных: {os.path.basename(file_path)}")
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', f'Ошибка подключения к БД: {str(e)}')

    def save_to_db(self):
        if self.df is not None:
            try:
                file_path, _ = QFileDialog.getSaveFileName(self, 'Сохранить в БД', '', 'SQLite Files (*.db)')
                if file_path:
                    conn = sqlite3.connect(file_path)
                    table_name = 'dataset'
                    self.df.to_sql(table_name, conn, if_exists='replace', index=False)
                    conn.close()
                    self.add_log(f"💾 Данные сохранены в БД: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, 'Ошибка', f'Ошибка сохранения в БД: {str(e)}')

    def update_table_list(self):
        if self.db_connection:
            cursor = self.db_connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            self.table_combo.clear()
            for table in tables:
                self.table_combo.addItem(table[0])

    def load_table_data(self, table_name):
        if self.db_connection and table_name:
            try:
                self.df = pd.read_sql_query(f"SELECT * FROM {table_name}", self.db_connection)
                self.current_table = table_name
                self.update_data_display()
                self.add_log(f"📊 Загружена таблица: {table_name}")
            except Exception as e:
                QMessageBox.critical(self, 'Ошибка', f'Ошибка загрузки таблицы: {str(e)}')

    def update_data_display(self):
        if self.df is not None:
            # Обновление информации
            info_text = f"📊 Данные: {self.df.shape[0]} строк × {self.df.shape[1]} столбцов | "
            info_text += f"📅 Числовых столбцов: {len(self.df.select_dtypes(include=[np.number]).columns)} | "
            info_text += f"🔤 Текстовых столбцов: {len(self.df.select_dtypes(include=['object']).columns)}"
            self.info_label.setText(info_text)

            # Обновление статистики
            stats_df = self.df.describe(include='all').T
            self.stats_table.setRowCount(len(stats_df))
            self.stats_table.setColumnCount(len(stats_df.columns) + 1)
            headers = ['Столбец'] + list(stats_df.columns)
            self.stats_table.setHorizontalHeaderLabels(headers)

            for i, (col_name, row) in enumerate(stats_df.iterrows()):
                self.stats_table.setItem(i, 0, QTableWidgetItem(str(col_name)))
                for j, value in enumerate(row):
                    self.stats_table.setItem(i, j + 1, QTableWidgetItem(str(value)))

            # Обновление отображения данных
            self.data_table.setRowCount(min(100, len(self.df)))
            self.data_table.setColumnCount(len(self.df.columns))
            self.data_table.setHorizontalHeaderLabels(self.df.columns.tolist())

            for i in range(min(100, len(self.df))):
                for j in range(len(self.df.columns)):
                    self.data_table.setItem(i, j, QTableWidgetItem(str(self.df.iloc[i, j])))

            # Обновление выпадающих списков
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.column_combo.clear()
            for col in numeric_columns:
                self.column_combo.addItem(col)

    def plot_correlation(self):
        if self.df is not None:
            try:
                # Очистка предыдущих графиков
                for i in reversed(range(self.corr_layout.count())):
                    widget = self.corr_layout.itemAt(i).widget()
                    if widget is not None:
                        widget.deleteLater()

                numeric_df = self.df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) < 2:
                    QMessageBox.warning(self, 'Предупреждение', 'Недостаточно числовых столбцов для анализа корреляции')
                    return

                # Построение парных графиков
                g = sns.pairplot(numeric_df)
                g.fig.suptitle('Парные графики корреляции', y=1.02)
                g.fig.set_size_inches(12, 10)

                # Сохранение графика в виджет
                figure = g.fig
                canvas = FigureCanvas(figure)
                self.corr_layout.addWidget(canvas)

                self.add_log("📊 Построены графики корреляции")

            except Exception as e:
                QMessageBox.critical(self, 'Ошибка', f'Ошибка построения графиков корреляции: {str(e)}')
                self.add_log(f"❌ Ошибка графиков корреляции: {str(e)}")

    def plot_heatmap(self):
        if self.df is not None:
            try:
                self.heatmap_figure.clear()
                numeric_df = self.df.select_dtypes(include=[np.number])

                if len(numeric_df.columns) < 2:
                    QMessageBox.warning(self, 'Предупреждение', 'Недостаточно числовых столбцов для тепловой карты')
                    return

                ax = self.heatmap_figure.add_subplot(111)
                correlation_matrix = numeric_df.corr()

                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')
                ax.set_title('Тепловая карта корреляций')

                self.heatmap_canvas.draw()
                self.add_log("🔥 Построена тепловая карта корреляций")

            except Exception as e:
                QMessageBox.critical(self, 'Ошибка', f'Ошибка построения тепловой карты: {str(e)}')
                self.add_log(f"❌ Ошибка тепловой карты: {str(e)}")

    def plot_line_chart(self):
        if self.df is not None and self.column_combo.currentText():
            try:
                self.line_figure.clear()
                column = self.column_combo.currentText()

                ax = self.line_figure.add_subplot(111)
                ax.plot(self.df.index, self.df[column], linewidth=2, color='blue', alpha=0.7)
                ax.set_title(f'Линейный график: {column}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Индекс')
                ax.set_ylabel(column)
                ax.grid(True, alpha=0.3)

                self.line_canvas.draw()
                self.add_log(f"📉 Построен линейный график для столбца: {column}")

            except Exception as e:
                QMessageBox.critical(self, 'Ошибка', f'Ошибка построения линейного графика: {str(e)}')
                self.add_log(f"❌ Ошибка линейного графика: {str(e)}")


def main():
    app = QApplication(sys.argv)
    window = DataVisualizationApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()