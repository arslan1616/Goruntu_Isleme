import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QInputDialog, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox
from PyQt5.QtGui import QPixmap, QImage, QTransform, QColor
from PyQt5.QtCore import Qt

class ImageProcessingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Ana pencere düzeni
        main_layout = QHBoxLayout()
        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Sol taraf - Butonlar
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout)

        self.assignment1_button = QPushButton('Görüntü Ekleme', self)
        left_layout.addWidget(self.assignment1_button)
        self.assignment1_button.clicked.connect(self.showAssignment1)

        self.assignment2_button = QPushButton('ÖDEV 2 ', self)
        left_layout.addWidget(self.assignment2_button)
        self.assignment2_button.clicked.connect(self.showAssignment2)

        # Sağ taraf - Görüntü ve başlık
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout)

        self.title_label = QLabel('Dijital Görüntü İşleme - 211229048 İbrahim Arslan', self)
        right_layout.addWidget(self.title_label, alignment=Qt.AlignCenter)

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(400, 400)
        right_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        self.pixel_value_label = QLabel(self)
        right_layout.addWidget(self.pixel_value_label, alignment=Qt.AlignCenter)

        # Ödev 1 menüsü
        self.assignment1_menu = QWidget(self)
        self.assignment1_menu.hide()
        right_layout.addWidget(self.assignment1_menu)

        assignment1_layout = QVBoxLayout(self.assignment1_menu)

        self.load_image_button = QPushButton('Görüntü Yükle', self.assignment1_menu)
        assignment1_layout.addWidget(self.load_image_button)
        self.load_image_button.clicked.connect(self.openImageDialog)

        self.back_button = QPushButton('Geri', self.assignment1_menu)
        assignment1_layout.addWidget(self.back_button)
        self.back_button.clicked.connect(self.showMainMenu)

        # Ödev 2 menüsü
        self.assignment2_menu = QWidget(self)
        self.assignment2_menu.hide()
        right_layout.addWidget(self.assignment2_menu)

        assignment2_layout = QVBoxLayout(self.assignment2_menu)

        self.enlarge_button = QPushButton('Görüntüyü Büyüt', self.assignment2_menu)
        assignment2_layout.addWidget(self.enlarge_button)
        self.enlarge_button.clicked.connect(self.enlargeImage)

        self.shrink_button = QPushButton('Görüntüyü Küçült', self.assignment2_menu)
        assignment2_layout.addWidget(self.shrink_button)
        self.shrink_button.clicked.connect(self.shrinkImage)

        self.zoom_in_button = QPushButton('Yakınlaştır', self.assignment2_menu)
        assignment2_layout.addWidget(self.zoom_in_button)
        self.zoom_in_button.clicked.connect(self.zoomIn)

        self.zoom_out_button = QPushButton('Uzaklaştır', self.assignment2_menu)
        assignment2_layout.addWidget(self.zoom_out_button)
        self.zoom_out_button.clicked.connect(self.zoomOut)

        self.rotate_button = QPushButton('Döndür', self.assignment2_menu)
        assignment2_layout.addWidget(self.rotate_button)
        self.rotate_button.clicked.connect(self.rotateImage)

        self.interpolation_button = QPushButton('Interpolasyon', self.assignment2_menu)
        assignment2_layout.addWidget(self.interpolation_button)
        self.interpolation_button.clicked.connect(self.applyInterpolation)

        self.back_button2 = QPushButton('Geri', self.assignment2_menu)
        assignment2_layout.addWidget(self.back_button2)
        self.back_button2.clicked.connect(self.showMainMenu)

    def showMainMenu(self):
        self.assignment1_menu.hide()
        self.assignment2_menu.hide()
        self.image_label.clear()
        self.pixel_value_label.clear()

    def showAssignment1(self):
        self.assignment1_menu.show()
        self.assignment2_menu.hide()

    def showAssignment2(self):
        self.assignment1_menu.hide()
        self.assignment2_menu.show()

    def openImageDialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Görüntü Aç", "",
                                                   "Görüntü Dosyaları (*.png *.jpg *.bmp)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

    def enlargeImage(self):
        try:
            factor, ok = QInputDialog.getDouble(self, "Büyütme Faktörü", "Faktör:", 2, 1, 10, 2)
            if ok:
                pixmap = self.image_label.pixmap()
                if not pixmap.isNull():
                    enlarged_pixmap = pixmap.scaled(pixmap.width() * factor, pixmap.height() * factor, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.image_label.setPixmap(enlarged_pixmap)
                else:
                    QMessageBox.warning(self, "Hata", "Görüntü yüklenmemiş.")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Bir hata oluştu: {str(e)}")

    def shrinkImage(self):
        try:
            factor, ok = QInputDialog.getDouble(self, "Küçültme Faktörü", "Faktör:", 0.5, 0.1, 1, 2)
            if ok:
                pixmap = self.image_label.pixmap()
                if not pixmap.isNull():
                    shrunk_pixmap = pixmap.scaled(pixmap.width() * factor, pixmap.height() * factor, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.image_label.setPixmap(shrunk_pixmap)
                else:
                    QMessageBox.warning(self, "Hata", "Görüntü yüklenmemiş.")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Bir hata oluştu: {str(e)}")

    def zoomIn(self):
        try:
            factor, ok = QInputDialog.getDouble(self, "Yakınlaştırma Faktörü", "Faktör:", 1.5, 1, 5, 2)
            if ok:
                pixmap = self.image_label.pixmap()
                if not pixmap.isNull():
                    zoomed_pixmap = pixmap.scaled(pixmap.width() * factor, pixmap.height() * factor, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.image_label.setPixmap(zoomed_pixmap)
                else:
                    QMessageBox.warning(self, "Hata", "Görüntü yüklenmemiş.")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Bir hata oluştu: {str(e)}")

    def zoomOut(self):
        try:
            factor, ok = QInputDialog.getDouble(self, "Uzaklaştırma Faktörü", "Faktör:", 0.5, 0.1, 1, 2)
            if ok:
                pixmap = self.image_label.pixmap()
                if not pixmap.isNull():
                    zoomed_pixmap = pixmap.scaled(pixmap.width() * factor, pixmap.height() * factor, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.image_label.setPixmap(zoomed_pixmap)
                else:
                    QMessageBox.warning(self, "Hata", "Görüntü yüklenmemiş.")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Bir hata oluştu: {str(e)}")

    def rotateImage(self):
        try:
            angle, ok = QInputDialog.getDouble(self, "Döndürme Açısı", "Açı (derece):", 0, -360, 360, 2)
            if ok:
                pixmap = self.image_label.pixmap()
                if not pixmap.isNull():
                    transform = QTransform().rotate(angle)
                    rotated_pixmap = pixmap.transformed(transform, Qt.SmoothTransformation)
                    self.image_label.setPixmap(rotated_pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                else:
                    QMessageBox.warning(self, "Hata", "Görüntü yüklenmemiş.")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Bir hata oluştu: {str(e)}")

    def applyInterpolation(self):
        try:
            interpolation_methods = ['Bilinear', 'Average']
            selected_method, ok = QInputDialog.getItem(self, "Interpolasyon Yöntemi", "Yöntem Seçin:", interpolation_methods, 0, False)
            if ok:
                pixmap = self.image_label.pixmap()
                if not pixmap.isNull():
                    image = pixmap.toImage()
                    width, height = image.width(), image.height()
                    scaled_image = QImage(width, height, QImage.Format_RGB32)

                    for y in range(height):
                        for x in range(width):
                            if selected_method == 'Bilinear':
                                scaled_image.setPixel(x, y, self.bilinearInterpolation(image, x, y))
                            elif selected_method == 'Average':
                                scaled_image.setPixel(x, y, self.averageInterpolation(image, x, y))

                    scaled_pixmap = QPixmap.fromImage(scaled_image)
                    self.image_label.setPixmap(scaled_pixmap)
                else:
                    QMessageBox.warning(self, "Hata", "Görüntü yüklenmemiş.")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Bir hata oluştu: {str(e)}")

    def bilinearInterpolation(self, image, x, y):
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, image.width() - 1), min(y0 + 1, image.height() - 1)
        dx, dy = x - x0, y - y0

        f00 = QColor(image.pixel(x0, y0)).getRgb()[:3]
        f10 = QColor(image.pixel(x1, y0)).getRgb()[:3]
        f01 = QColor(image.pixel(x0, y1)).getRgb()[:3]
        f11 = QColor(image.pixel(x1, y1)).getRgb()[:3]

        r = (1 - dx) * (1 - dy) * f00[0] + dx * (1 - dy) * f10[0] + (1 - dx) * dy * f01[0] + dx * dy * f11[0]
        g = (1 - dx) * (1 - dy) * f00[1] + dx * (1 - dy) * f10[1] + (1 - dx) * dy * f01[1] + dx * dy * f11[1]
        b = (1 - dx) * (1 - dy) * f00[2] + dx * (1 - dy) * f10[2] + (1 - dx) * dy * f01[2] + dx * dy * f11[2]

        return QColor(int(r), int(g), int(b)).rgb()

    def averageInterpolation(self, image, x, y):
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, image.width() - 1), min(y0 + 1, image.height() - 1)

        f00 = QColor(image.pixel(x0, y0)).getRgb()[:3]
        f10 = QColor(image.pixel(x1, y0)).getRgb()[:3]
        f01 = QColor(image.pixel(x0, y1)).getRgb()[:3]
        f11 = QColor(image.pixel(x1, y1)).getRgb()[:3]

        r = (f00[0] + f10[0] + f01[0] + f11[0]) / 4
        g = (f00[1] + f10[1] + f01[1] + f11[1]) / 4
        b = (f00[2] + f10[2] + f01[2] + f11[2]) / 4

        return QColor(int(r), int(g), int(b)).rgb()

    def mouseMoveEvent(self, event):
        if not self.image_label.pixmap().isNull():
            x = event.x() - self.image_label.x()
            y = event.y() - self.image_label.y()
            pixmap = self.image_label.pixmap()
            if 0 <= x < pixmap.width() and 0 <= y < pixmap.height():
                image = pixmap.toImage()
                pixel_value = image.pixel(x, y)
                color = QColor(pixel_value)
                self.pixel_value_label.setText(f"Piksel Değeri: ({x}, {y}) = {color.name()}")
            else:
                self.pixel_value_label.clear()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = ImageProcessingGUI()
    gui.show()
    sys.exit(app.exec_())