import sys
import cv2
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QInputDialog, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox
from PyQt5.QtGui import QPixmap, QImage, QTransform, QColor
from PyQt5.QtCore import Qt
from skimage.measure import label, regionprops

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

        self.assignment2_button = QPushButton('ÖDEV 2', self)
        left_layout.addWidget(self.assignment2_button)
        self.assignment2_button.clicked.connect(self.showAssignment2)

        self.midterm_button = QPushButton('Vize Ödevi', self)
        left_layout.addWidget(self.midterm_button)
        self.midterm_button.clicked.connect(self.showMidtermAssignment)

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

        # Vize Ödevi menüsü
        self.midterm_menu = QWidget(self)
        self.midterm_menu.hide()
        right_layout.addWidget(self.midterm_menu)

        midterm_layout = QVBoxLayout(self.midterm_menu)

        self.sigmoid_button = QPushButton('Sigmoid', self.midterm_menu)
        midterm_layout.addWidget(self.sigmoid_button)
        self.sigmoid_button.clicked.connect(self.applySigmoid)

        self.hough_button = QPushButton('Hough Transform', self.midterm_menu)
        midterm_layout.addWidget(self.hough_button)
        self.hough_button.clicked.connect(self.applyHoughTransform)

        self.deblurring_button = QPushButton('Deblurring', self.midterm_menu)
        midterm_layout.addWidget(self.deblurring_button)
        self.deblurring_button.clicked.connect(self.applyDeblurring)

        self.object_detection_button = QPushButton('Nesne Tanımlama', self.midterm_menu)
        midterm_layout.addWidget(self.object_detection_button)
        self.object_detection_button.clicked.connect(self.applyObjectDetection)

        self.back_button3 = QPushButton('Geri', self.midterm_menu)
        midterm_layout.addWidget(self.back_button3)
        self.back_button3.clicked.connect(self.showMainMenu)

    def showMainMenu(self):
        self.assignment1_menu.hide()
        self.assignment2_menu.hide()
        self.midterm_menu.hide()
        self.image_label.clear()
        self.pixel_value_label.clear()

    def showAssignment1(self):
        self.assignment1_menu.show()
        self.assignment2_menu.hide()
        self.midterm_menu.hide()

    def showAssignment2(self):
        self.assignment1_menu.hide()
        self.assignment2_menu.show()
        self.midterm_menu.hide()

    def showMidtermAssignment(self):
        self.assignment1_menu.hide()
        self.assignment2_menu.hide()
        self.midterm_menu.show()

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
                    enlarged_pixmap = pixmap.scaled(int(pixmap.width() * factor), int(pixmap.height() * factor), Qt.KeepAspectRatio, Qt.SmoothTransformation)
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
                    shrunk_pixmap = pixmap.scaled(int(pixmap.width() * factor), int(pixmap.height() * factor), Qt.KeepAspectRatio, Qt.SmoothTransformation)
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
                    zoomed_pixmap = pixmap.scaled(int(pixmap.width() * factor), int(pixmap.height() * factor), Qt.KeepAspectRatio, Qt.SmoothTransformation)
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
                    zoomed_pixmap = pixmap.scaled(int(pixmap.width() * factor), int(pixmap.height() * factor), Qt.KeepAspectRatio, Qt.SmoothTransformation)
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

        r = int((1 - dx) * (1 - dy) * f00[0] + dx * (1 - dy) * f10[0] + (1 - dx) * dy * f01[0] + dx * dy * f11[0])
        g = int((1 - dx) * (1 - dy) * f00[1] + dx * (1 - dy) * f10[1] + (1 - dx) * dy * f01[1] + dx * dy * f11[1])
        b = int((1 - dx) * (1 - dy) * f00[2] + dx * (1 - dy) * f10[2] + (1 - dx) * dy * f01[2] + dx * dy * f11[2])

        return QColor(r, g, b).rgb()

    def averageInterpolation(self, image, x, y):
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, image.width() - 1), min(y0 + 1, image.height() - 1)
        f00 = QColor(image.pixel(x0, y0)).getRgb()[:3]
        f10 = QColor(image.pixel(x1, y0)).getRgb()[:3]
        f01 = QColor(image.pixel(x0, y1)).getRgb()[:3]
        f11 = QColor(image.pixel(x1, y1)).getRgb()[:3]
        r = int((f00[0] + f10[0] + f01[0] + f11[0]) / 4)
        g = int((f00[1] + f10[1] + f01[1] + f11[1]) / 4)
        b = int((f00[2] + f10[2] + f01[2] + f11[2]) / 4)
    
        return QColor(r, g, b).rgb()

    def applySigmoid(self):
        try:
            sigmoid_functions = ['Standart Sigmoid', 'Yatay Kaydırılmış Sigmoid', 'Eğimli Sigmoid', 'Kendi Fonksiyonum']
            selected_function, ok = QInputDialog.getItem(self, "Sigmoid Fonksiyonu", "Fonksiyon Seçin:", sigmoid_functions, 0, False)
            if ok:
                pixmap = self.image_label.pixmap()
                if not pixmap.isNull():
                    image = pixmap.toImage()
                    width, height = image.width(), image.height()
                    processed_image = QImage(width, height, QImage.Format_RGB32)
    
                    for y in range(height):
                        for x in range(width):
                            pixel = image.pixel(x, y)
                            r, g, b = QColor(pixel).getRgb()[:3]
                            gray = (r + g + b) // 3
    
                            if selected_function == 'Standart Sigmoid':
                                processed_gray = self.standardSigmoid(gray)
                            elif selected_function == 'Yatay Kaydırılmış Sigmoid':
                                processed_gray = self.horizontallyShiftedSigmoid(gray)
                            elif selected_function == 'Eğimli Sigmoid':
                                processed_gray = self.slopedSigmoid(gray)
                            elif selected_function == 'Kendi Fonksiyonum':
                                processed_gray = self.customSigmoid(gray)
    
                            processed_pixel = QColor(processed_gray, processed_gray, processed_gray).rgb()
                            processed_image.setPixel(x, y, processed_pixel)
    
                    processed_pixmap = QPixmap.fromImage(processed_image)
                    self.image_label.setPixmap(processed_pixmap)
                else:
                    QMessageBox.warning(self, "Hata", "Görüntü yüklenmemiş.")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Bir hata oluştu: {str(e)}")
    
    def standardSigmoid(self, x):
        return int(255 / (1 + np.exp(-0.05 * (x - 128))))
    
    def horizontallyShiftedSigmoid(self, x):
        return int(255 / (1 + np.exp(-0.05 * (x - 160))))
    
    def slopedSigmoid(self, x):
        return int(255 / (1 + np.exp(-0.02 * (x - 128))))
    
    def customSigmoid(self, x):
        return int(255 / (1 + np.exp(-0.03 * (x - 96))))

    def applyHoughTransform(self):
           options = ['Yoldaki Çizgileri Tespit Et', 'Yüz Resminde Gözleri Tespit Et']
           selected_option, ok = QInputDialog.getItem(self, "Hough Transform", "Seçenek:", options, 0, False)
           if ok:
            pixmap = self.image_label.pixmap()
            if not pixmap.isNull():
                image = pixmap.toImage()
                width, height = image.width(), image.height()
                bytes_per_line = image.bytesPerLine()
                image_data = image.bits().asstring(bytes_per_line * height)
                cv_image = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 4))
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2BGR)
    
                if selected_option == 'Yoldaki Çizgileri Tespit Et':
                    self.detectRoadLines(cv_image)
                elif selected_option == 'Yüz Resminde Gözleri Tespit Et':
                    self.detectEyesInFace(cv_image)
            else:
                QMessageBox.warning(self, "Hata", "Görüntü yüklenmemiş.")
    def detectRoadLines(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = processed_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(processed_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image))
                
    def detectEyesInFace(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('C:/Users/Power/classifiers/haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('C:/Users/Power/classifiers/haarcascade_eye.xml')
    
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = processed_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(processed_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image))
    
    def applyDeblurring(self):
        pixmap = self.image_label.pixmap()
        if not pixmap.isNull():
            image = pixmap.toImage()
            width, height = image.width(), image.height()
            bytes_per_line = image.bytesPerLine()
            image_data = image.bits().asstring(bytes_per_line * height)
            cv_image = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 4))
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2BGR)
            
       
            size = 31
            kernel_motion_blur = np.zeros((size, size))
            kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
            kernel_motion_blur = kernel_motion_blur / size
            
        
            deblurred_image = self.wiener_filter(cv_image, kernel_motion_blur, K=0.001)
            
            processed_image = cv2.cvtColor(deblurred_image, cv2.COLOR_BGR2RGB)
            height, width, channel = processed_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(processed_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_image))
        else:
            QMessageBox.warning(self, "Hata", "Görüntü yüklenmemiş.")

    def wiener_filter(self, img, kernel, K):
        kernel_padded = np.pad(kernel, [
            (0, img.shape[0] - kernel.shape[0]), 
            (0, img.shape[1] - kernel.shape[1])
        ], 'constant')
        
       
        kernel_fft = np.fft.fft2(kernel_padded)
        kernel_conj = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + K)
        
        img_filtered = np.zeros_like(img, dtype=float)
        for i in range(3):  # RGB kanalları için döngü
            img_fft = np.fft.fft2(img[:, :, i])
            img_deblurred_fft = img_fft * kernel_conj
            img_deblurred = np.abs(np.fft.ifft2(img_deblurred_fft))
            img_filtered[:, :, i] = img_deblurred
        
        
        img_filtered = np.clip(img_filtered, 0, 255)
        return img_filtered.astype(np.uint8)
            
    def applyObjectDetection(self):
        pixmap = self.image_label.pixmap()
        if not pixmap.isNull():
            image = pixmap.toImage()
            width, height = image.width(), image.height()
            bytes_per_line = image.bytesPerLine()
            image_data = image.bits().asstring(bytes_per_line * height)
            cv_image = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 4))
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2BGR)
    
            # Yeşil bölgeleri tespit et ve özellikleri hesapla
            self.detectGreenRegions(cv_image)
        else:
            QMessageBox.warning(self, "Hata", "Görüntü yüklenmemiş.")
    def detectGreenRegions(self, cv_image):
        # Görüntüyü HSV renk uzayına dönüştür
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    
        # Yeşil tonlarını filtrele
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    
        
        labeled_image = label(green_mask)
    
        
        regions = regionprops(labeled_image, intensity_image=cv_image)
    
      
        data = []
        for region in regions:
            bbox = region.bbox
            centroid = region.centroid
            length = bbox[2] - bbox[0]
            width = bbox[3] - bbox[1]
            diagonal = np.sqrt(length**2 + width**2)
            energy = np.sum(region.intensity_image)
            entropy = self.calculate_entropy(region.intensity_image)
            mean_intensity = np.mean(region.intensity_image)
            median_intensity = np.median(region.intensity_image)
    
            data.append([
                region.label,
                centroid[1], centroid[0],
                length, width, diagonal,
                energy, entropy,
                mean_intensity, median_intensity
            ])
    
      
        df = pd.DataFrame(data, columns=['No', 'Center X', 'Center Y', 'Length', 'Width', 'Diagonal', 'Energy', 'Entropy', 'Mean', 'Median'])
    
        
        df.to_excel('yesilbölge.xlsx', index=False)
    
        QMessageBox.information(self, "Bilgi", "Yeşil bölgeler tespit edildi ve özellikleri 'yesilbölge.xlsx' dosyasına kaydedildi.")
        
        QMessageBox.information(self, "Bilgi", "Yeşil bölgeler tespit edildi ve özellikleri 'green_regions.xlsx' dosyasına kaydedildi.")
    def calculate_entropy(self, image):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        logs = np.log2(hist + 1e-7)
        entropy = -np.sum(hist * logs)
        return entropy

                
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