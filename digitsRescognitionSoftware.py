from PyQt5 import QtCore, QtGui, QtWidgets,uic
from PyQt5.QtWidgets import QApplication, QWidget, QLabel,QPushButton,QMessageBox,QInputDialog,QLineEdit,QMainWindow, QLabel,QGridLayout, QWidget, QDesktopWidget,QFileDialog,QStatusBar
from PyQt5.QtGui import QIcon, QPixmap,QImage,QPainter,QColor
from PyQt5.QtCore import pyqtSlot,QThread,pyqtSignal,Qt,QTimer
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from datetime import date
from threading import Timer,Thread,Event
import tensorflow as tf
import time
import datetime
import sys
import os
import numpy as np
import keras
import keras_retinanet
import cv2
import urllib.request
import subprocess
import time
import win32gui
import threading

global graph,model

def get_session():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	return tf.Session(config=config)

graph = tf.get_default_graph()

keras.backend.tensorflow_backend.set_session(get_session())

model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.load_weights('vgg16MnistModel.h5', by_name = True)

class Thread1(QThread):
	
	flag_IP_start_display_1 = -1
	flag_skip_display_1_save = -1
	frame_global_display_1 = []
	feed = ''
	filename = ''
	val = -1
	
	def get_feed(self,feed):
		self.feed = feed

	def get_filename(self,fileName):
		self.filename = fileName

	changePixmap = pyqtSignal(QImage)
	valueSignaldisplay1 = pyqtSignal(str)
	
	def run(self):
				
		with graph.as_default():

				head = time.time()
				
				if self.flag_IP_start_display_1 == 1:

					while self.flag_IP_start_display_1 != -1:
						
						try:
							self.val = -1
							url = "http://"+str(self.feed)+"/shot.jpg"
							print(url)                    							
							imgResp = urllib.request.urlopen(url)                               					
							tail = time.time() - head
							#print("Time taken to read "+str(frame_10_sec)+" frames", tail)
							start = time.time()                        
							imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
							self.frame_global_display_1 = imgNp
							image = cv2.imdecode(imgNp,-1)
							image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
							image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
							imgPred = image.reshape(1, 28, 28, 1)
							imgPred = imgPred.astype('float32')
                            imgPred = imgPred / 255.0
							pred = model.predict(imgPred)							
							
							pred = pred.argmax()
								
							if (pred == 0):
								self.val = 0
							
							elif (pred == 1):
								self.val = 1

							elif (pred == 2):
								self.val = 2

							elif (pred == 3):
								self.val = 3

							elif (pred == 4):
								self.val = 4

							elif (pred == 5):
								self.val = 5

							elif (pred == 6):
								self.val = 6

							elif (pred == 7):
								self.val = 7

							elif (pred == 8):
								self.val = 8

							else:
								self.val = 9

							self.flag_skip_display_1_save = 0		   
							rgbImage = image						
							height, width,channel = np.shape(image)
							totalBytes = image.nbytes
							bytesPerLine = int(totalBytes/height)
							convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0],bytesPerLine, QImage.Format_RGB888)
							p = convertToQtFormat.scaled(715*s_w, 350*s_l, QtCore.Qt.KeepAspectRatio)
							self.changePixmap.emit(p)
							self.valueSignaldisplay1.emit(str(self.val))
							var_no = ""
							self.NoerrorSignaldisplay1.emit(var_no)
							print("processing time screen 1", time.time() - start)
							print("Total Lag of screen 1", tail + (time.time() - start))
							head = time.time()
												
						except:
							self.flag_skip_display_1_save = 1
							no_feed_image = cv2.imread("no_connection.jpg")
							no_feed_image = cv2.cvtColor(no_feed_image, cv2.COLOR_BGR2RGB)    
							convertToQtFormat = QImage(no_feed_image.data, no_feed_image.shape[1], no_feed_image.shape[0], QImage.Format_RGB888)
							p = convertToQtFormat.scaled(715*s_w, 350*s_l, QtCore.Qt.KeepAspectRatio)
							self.changePixmap.emit(p) 
							var = "Error : No Feed Found on {}".format(str(self.feed))
							self.errorSignaldisplay1.emit(var)
						
			
					self.flag_skip_display_1_save = 1

				elif self.flag_IP_start_display_1 == 0:
				
					count_frame_no = 0
					
					try:

						cap = cv2.VideoCapture(str(self.filename))
						print("video loaded in cap")
					
						while cap.isOpened():
							
							self.val = -1

							ret,frame = cap.read()
							print("Cap reads frame")
							
							if ret:
								
								var_no = ""
								self.NoerrorSignaldisplay1.emit(var_no)
								count_frame_no = count_frame_no + 1
								print(count_frame_no)
								
								if(count_frame_no % 20 == 0):
						
									start = time.time()
									image = frame
									self.frame_global_display_1 = frame
									image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
									image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
									imgPred = image.reshape(1, 28, 28, 1)
									imgPred = imgPred.astype('float32')
	                                imgPred = imgPred / 255.0
									pred = model.predict(imgPred)						
									
									pred = pred.argmax()
										
									if (pred == 0):
										self.val = 0
									
									elif (pred == 1):
										self.val = 1

									elif (pred == 2):
										self.val = 2

									elif (pred == 3):
										self.val = 3

									elif (pred == 4):
										self.val = 4

									elif (pred == 5):
										self.val = 5

									elif (pred == 6):
										self.val = 6

									elif (pred == 7):
										self.val = 7

									elif (pred == 8):
										self.val = 8

									else:
										self.val = 9
										
									self.flag_skip_display_1_save = 0

									rgbImage = image     
									
									height, width,channel = np.shape(image)
									totalBytes = image.nbytes
									bytesPerLine = int(totalBytes/height)
									
									convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], bytesPerLine, QImage.Format_RGB888)
									p = convertToQtFormat.scaled(715*s_w, 350*s_l, QtCore.Qt.KeepAspectRatio)
									self.changePixmap.emit(p)
									self.valueSignaldisplay1.emit(str(self.val))
									count_frame_no = 0
									print("processing time", time.time() - start)
							
							if(self.flag_IP_start_display_1 == -1):
								print("Video Stopped")    
								self.flag_skip_display_1_save = 1
								cap.release()
						
						self.flag_skip_display_1_save = 1

					except:
						self.flag_skip_display_1_save = 1
						no_feed_image = cv2.imread("no_connection.jpg")
						no_feed_image = cv2.cvtColor(no_feed_image, cv2.COLOR_BGR2RGB)    
						convertToQtFormat = QImage(no_feed_image.data, no_feed_image.shape[1], no_feed_image.shape[0], QImage.Format_RGB888)
						p = convertToQtFormat.scaled(715*s_w, 350*s_l, QtCore.Qt.KeepAspectRatio)
						self.changePixmap.emit(p) 
						var = "Error : No Feed Found on {}".format(str(self.feed))
						self.errorSignaldisplay1.emit(var)

class Ui_MainWindow(object):
	
	def __init__(self,MainWindow):
	
		MainWindow.setObjectName("MainWindow")
		MainWindow.resize(1245*s_w, 680*s_l)
		MainWindow.setMaximumSize(QtCore.QSize(1245*s_w, 680*s_l))
		MainWindow.setGeometry(QtCore.QRect(70*s_w, 60*s_l, 1245*s_w, 680*s_l))
		MainWindow.setStyleSheet("background-color: rgb(31, 31, 31);\n"
	"color: rgb(255, 255, 255);\n"
	"")

		self.centralwidget = QtWidgets.QWidget(MainWindow)
		self.centralwidget.setObjectName("centralwidget")

		icon = QtGui.QIcon()
		icon.addPixmap(QtGui.QPixmap('crop.png'),QtGui.QIcon.Normal,QtGui.QIcon.Off)
		MainWindow.setWindowIcon(icon)
		self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
		self.tabWidget.setGeometry(QtCore.QRect(0, 0, 1245*s_w, 680*s_l))
		self.tabWidget.setStyleSheet("background-color: rgb(31, 31, 31);\n""color: rgb(0, 0, 0);\n")
		self.tabWidget.setObjectName("tabWidget")
	
class Tab():

	def __init__(self,object_name):
		self.th = object_name

	def setupUi(self,MainWindow,tabWidget,centralwidget):
		
		self.centralwidget = centralwidget
		self.tabWidget = tabWidget
		self.tab = QtWidgets.QTabWidget()
		self.tab.setObjectName("tab")		
		self.tabWidget.addTab(self.tab, "")		
		self.horizontalLayoutWidget = QtWidgets.QWidget(self.tab)
		self.horizontalLayoutWidget.setGeometry(QtCore.QRect(250*s_w, 20*s_l, 715*s_w, 350*s_l))
		self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
		self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
		self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
		self.horizontalLayout.setObjectName("horizontalLayout")
		self.Screen_1 = QtWidgets.QLabel(self.horizontalLayoutWidget)
		self.Screen_1.setAlignment(Qt.AlignCenter)
		self.Screen_1.setStyleSheet("background-color: rgb(0, 0, 0);")
		self.Screen_1.setObjectName("Screen_1")
		self.horizontalLayout.addWidget(self.Screen_1)
		self.start_screen_1 = QtWidgets.QPushButton(self.tab)
		self.start_screen_1.setGeometry(QtCore.QRect(335*s_w, 380*s_l, 75*s_w, 23*s_l))
		self.start_screen_1.setStyleSheet("background-color: rgb(0, 0, 0);\n"
"color: rgb(255, 255, 255);")
		self.start_screen_1.setObjectName("start_screen_1")
		self.start_screen_1.clicked.connect(self.on_click_start_display_1)
		self.video_screen_1 = QtWidgets.QPushButton(self.tab)
		self.video_screen_1.setGeometry(QtCore.QRect(485*s_w, 380*s_l, 75*s_w, 23*s_l))
		self.video_screen_1.setStyleSheet("background-color: rgb(0, 0, 0);\n"
"color: rgb(255, 255, 255);")
		self.video_screen_1.setObjectName("video_screen_1")
		self.video_screen_1.clicked.connect(self.on_click_start_vid_display_1)
		self.stop_screen_1 = QtWidgets.QPushButton(self.tab)
		self.stop_screen_1.setGeometry(QtCore.QRect(635*s_w, 380*s_l, 75*s_w, 23*s_l))
		self.stop_screen_1.setStyleSheet("background-color: rgb(0, 0, 0);\n"
"color: rgb(255, 255, 255);")
		self.stop_screen_1.setObjectName("stop_screen_1")
		self.stop_screen_1.clicked.connect(self.on_click_stop_rec_cam1_display_1)
		self.save_screen_1 = QtWidgets.QPushButton(self.tab)
		self.save_screen_1.setGeometry(QtCore.QRect(785*s_w, 380*s_l, 75*s_w, 23*s_l))
		self.save_screen_1.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: rgb(0, 0, 0);")
		self.save_screen_1.setObjectName("save_screen_1")
		self.save_screen_1.clicked.connect(self.on_click_save_display_1)       
		
		MainWindow.setCentralWidget(self.centralwidget)
		
		self.value_display_1 = QtWidgets.QLabel(self.tab)
		self.value_display_1.setGeometry(QtCore.QRect(500*s_w, 440*s_l, 103*s_w, 21*s_l))
		self.value_display_1.setStyleSheet("font: 14pt \"Kalinga\";""color :rgb(255,255,255)")
		self.value_display_1.setObjectName("value_display_1")
		
		self.value_alert_display_1 = QtWidgets.QLabel(self.tab)
		self.value_alert_display_1.setGeometry(QtCore.QRect(687*s_w, 440*s_l, 103*s_w, 21*s_l))
		self.value_alert_display_1.setStyleSheet("font: 14pt \"Kalinga\";""color :rgb(255,255,255)")
		self.value_alert_display_1.setObjectName("value_alert_display_1")
		
		self.error_display_1 = QtWidgets.QLabel(self.tab)
		self.error_display_1.setGeometry(QtCore.QRect(453*s_w, 590*s_l, 270*s_w, 23*s_l))
		self.error_display_1.setStyleSheet("font: 9pt \"Kalinga\";""color :rgb(255,255,255)")
		self.error_display_1.setObjectName("error_display_1")
		self.error_display_1.setAlignment(Qt.AlignCenter)

		self.menubar = QtWidgets.QMenuBar(MainWindow)
		self.menubar.setGeometry(QtCore.QRect(0, 0, 913*s_w, 21*s_l))
		self.menubar.setObjectName("menubar")
		self.menuFile = QtWidgets.QMenu(self.menubar)
		self.menuFile.setObjectName("menuFile")
		open_action = QtWidgets.QAction('Open Log Folder', self.menuFile)
		open_action.setShortcut('Ctrl+L')
		self.menuFile.addAction(open_action)
		open_action.triggered.connect(self.open_log_folder)
		self.menuEdit = QtWidgets.QMenu(self.menubar)
		self.menuEdit.setObjectName("menuEdit")
		self.menuAbout = QtWidgets.QMenu(self.menubar)
		self.menuAbout.setObjectName("menuAbout")
		self.menuHelp = QtWidgets.QMenu(self.menubar)
		self.menuHelp.setObjectName("menuHelp")
		MainWindow.setMenuBar(self.menubar)
		self.statusbar = QtWidgets.QStatusBar(MainWindow)
		self.statusbar.setObjectName("statusbar")
		MainWindow.setStatusBar(self.statusbar)
		self.menubar.addAction(self.menuFile.menuAction())
		self.menubar.addAction(self.menuEdit.menuAction())
		self.menubar.addAction(self.menuAbout.menuAction())
		self.menubar.addAction(self.menuHelp.menuAction())

		self.retranslateUi(MainWindow,self.tabWidget)
		self.tabWidget.setCurrentIndex(0)
		QtCore.QMetaObject.connectSlotsByName(MainWindow)

	def retranslateUi(self, MainWindow,tabWidget):
		_translate = QtCore.QCoreApplication.translate
		self.tabWidget = tabWidget
		MainWindow.setWindowTitle(_translate("MainWindow", "AIonAsset"))
		self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Display Screen"))
		self.Dtetection_lbl.setText(_translate("MainWindow", "Detections"))		
		self.Screen_1.setText(_translate("MainWindow", "                                                                                                            TextLabel"))
		self.start_screen_1.setText(_translate("MainWindow", "Start"))
		self.video_screen_1.setText(_translate("MainWindow", "Video"))
		self.stop_screen_1.setText(_translate("MainWindow", "Stop"))
		self.save_screen_1.setText(_translate("MainWindow", "Save"))
		self.value_display_1.setText(_translate("MainWindow", "Value"))
		self.value_alert_display_1.setText(_translate("MainWindow", ""))
		self.error_display_1.setText(_translate("MainWindow", ""))

		self.menuFile.setTitle(_translate("MainWindow", "File"))
		self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
		self.menuAbout.setTitle(_translate("MainWindow", "About"))
		self.menuHelp.setTitle(_translate("MainWindow", "Help"))

	def open_log_folder(self,MainWindow):
		os.system('explorer.exe "logs"')

	def get_project(self,MainWindow):
		
		proj_name, okPressed = QInputDialog.getText(self.centralwidget, "AIonAsset","Enter Project Name:", QLineEdit.Normal, "")

		if okPressed and proj_name != '':
			proj_name = str(proj_name)
			return(True,proj_name)
		
		else:
			return(False,'0')

	def get_cam_no(self,MainWindow):
		
		cam_no, okPressed = QInputDialog.getText(self.centralwidget, "AIonAsset","Enter Camera Number:", QLineEdit.Normal, "")

		if okPressed and cam_no != '':
			cam_no = str(cam_no)
			return(True,cam_no)
		
		else:
			return(False,'0')

	def openFileNameDialog(self,MainWindow): 
		
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		#fileName, _ = QFileDialog.getOpenFileName(self.centralwidget,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
		fileName, _ = QFileDialog.getOpenFileName(self.centralwidget,"QFileDialog.getOpenFileName()","","Video Files (*mp4 *.m4v *.wmv *.avi *.mov *.flv *.mpg *.3gp *.asf *.rm *.swf *.mkv);;All Files (*)", options=options)
		if fileName:
			return(True,fileName)
		else:
			return(False,fileName)  

	def auto_save_display_1_feed(self):
		
		if (self.th.flag_skip_display_1_save == 0):
			
			img = cv2.imdecode(self.th.frame_global_display_1,-1)
			rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
			date_new = str(datetime.date.today())
			a = str((datetime.datetime.now().time())).split(":")
			time_new = "Date - "+date_new+"- Time - "+a[0]+"-"+a[1]+"-"+a[2][0:4]
			cv2.imwrite("logs/saved_image/"+str(time_new)+".jpg",img)

	def auto_save_display_1_video(self):

		if (self.th.flag_skip_display_1_save == 0):
			
			rgbImage = cv2.cvtColor(self.th.frame_global_display_1, cv2.COLOR_BGR2RGB)
			img = np.asarray(self.th.frame_global_display_1)

			date_new = str(datetime.date.today())
			a = str((datetime.datetime.now().time())).split(":")
			time_new = "Date - "+date_new+"- Time - "+a[0]+"-"+a[1]+"-"+a[2][0:4]
			cv2.imwrite("logs/saved_image/"+str(time_new)+".jpg",img)

	def on_click_save_display_1(self,MainWindow):
		
		if(self.th.flag_skip_display_1_save == 0):

			if(self.th.flag_IP_start_display_1 == 1):
				img = cv2.imdecode(self.th.frame_global_display_1,-1)
				rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
				height, width,channel = np.shape(rgbImage)
				totalBytes = rgbImage.nbytes
				bytesPerLine = int(totalBytes/height)

			else:
				rgbImage = cv2.cvtColor(self.th.frame_global_display_1, cv2.COLOR_BGR2RGB)
				img = np.asarray(self.th.frame_global_display_1)
				height, width,channel = np.shape(img)
				totalBytes = img.nbytes
				bytesPerLine = int(totalBytes/height)

			date_new = str(datetime.date.today())
			a = str((datetime.datetime.now().time())).split(":")
			time_new = "Date - "+date_new+"- Time - "+a[0]+"-"+a[1]+"-"+a[2][0:4]
			cv2.imwrite("logs/saved_image/"+str(time_new)+".jpg",img)

	def on_click_stop_rec_cam1_display_1(self,MainWindow):

		print("Done")

		if (self.th.flag_IP_start_display_1 == 1):
			self.timer_feed_display_1.stop()

		if(self.th.flag_IP_start_display_1 == 0):
			self.timer_video_display_1.stop()

		self.th.flag_skip_display_1_save = 1
		self.th.flag_IP_start_display_1 = -1

		self.start_screen_1.setEnabled(True)
		self.video_screen_1.setEnabled(True)

	def setImage_display_1(self, image):
		self.Screen_1.setPixmap(QPixmap.fromImage(image))

	def alert_value_display1(self,string):
		self.value_alert_display_1.setText(string)

	def error_msg_display_1(self,string):
		self.error_display_1.setText(string)
	
	def no_error_msg_display_1(self,string):
		self.error_display_1.setText(string)

	def getUrl_display_1(self,MainWindow):
		
		Url, okPressed = QInputDialog.getText(self.centralwidget, "AIonAsset","Enter Cam 1 Url:", QLineEdit.Normal, "")

		if okPressed and Url != '':
			Url = str(Url)
			#self.th.feed = Url
			return(True,Url)
		
		else:
			return(False,0)

	def on_click_start_vid_display_1(self,MainWindow):

		file,filename = self.openFileNameDialog(self.centralwidget)

		if file:

			proj, proj_name = self.get_project(self.centralwidget)
				
			if proj:

				cam, cam_no = self.get_cam_no(self.centralwidget)

				if cam:
				
					self.timer_video_display_1 = QTimer(self.centralwidget)
					self.timer_video_display_1.timeout.connect(self.auto_save_display_1_video)
					self.timer_video_display_1.start(1 * 60 * 1000)
					self.th.get_filename(filename)
					self.th.valueSignaldisplay1.connect(self.alert_value_display1)
					self.th.NoerrorSignaldisplay1.connect(self.no_error_msg_display_1)
					self.th.changePixmap.connect(self.setImage_display_1)
					self.th.start()
					self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab),'1 - ' + proj_name + ' - ' + cam_no)
					self.video_screen_1.setEnabled(False)
					self.start_screen_1.setEnabled(False)
					self.save_screen_1.setEnabled(True)
					self.th.flag_IP_start_display_1 = 0

	def on_click_start_display_1(self,MainWindow):

		feed,Url = self.getUrl_display_1(self.centralwidget)

		if feed:

				proj, proj_name = self.get_project(self.centralwidget)
				
				if proj:

					cam, cam_no = self.get_cam_no(self.centralwidget)

					if cam:

						self.th.get_feed(Url)
						self.timer_feed_display_1 = QTimer(self.centralwidget)
						self.timer_feed_display_1.timeout.connect(self.auto_save_display_1_feed)
						self.timer_feed_display_1.start(1 * 1000 * 60)
						self.th.changePixmap.connect(self.setImage_display_1)
						self.th.valueSignaldisplay1.connect(self.alert_value_display1)
						self.th.errorSignaldisplay1.connect(self.error_msg_display_1) 
						self.th.NoerrorSignaldisplay1.connect(self.no_error_msg_display_1)
						self.th.start()
						self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab),'1 - ' + proj_name + ' - ' + cam_no)
						self.save_screen_1.setEnabled(True)
						self.start_screen_1.setEnabled(False)
						self.video_screen_1.setEnabled(False)
						self.th.flag_IP_start_display_1 = 1

				

if __name__ == "__main__":
	
	import sys

	app = QtWidgets.QApplication(sys.argv)

	screen = app.primaryScreen()
	print('Screen: %s' % screen.name())
	size = screen.size()
	print('Size: %d x %d' % (size.width(), size.height()))
	rect = screen.availableGeometry()
	print('Available: %d x %d' % (rect.width(), rect.height()))

	width_ratio = 1245 / 1366
	height_ratio = 650 / 768

	s_w = ((size.width() * width_ratio) / 1245) 
	print(s_w)
	
	s_l = ((size.height() * height_ratio) / 650)
	print(s_l)
	
	if(s_w < 1):
		s_w = 1

	if(s_l < 1):
		s_l = 1    
	
	CreateDataset_main_window = QtWidgets.QMainWindow()
	ui = Ui_MainWindow(CreateDataset_main_window)
	th1 = Thread1()
	tab1 = Tab(th1)
	tab1.setupUi(CreateDataset_main_window,ui.tabWidget,ui.centralwidget)
	th2 = Thread1()
	tab2 = Tab(th2)
	tab2.setupUi(CreateDataset_main_window,ui.tabWidget,ui.centralwidget)
	th3 = Thread1()
	tab3 = Tab(th3)
	tab3.setupUi(CreateDataset_main_window,ui.tabWidget,ui.centralwidget)
	th4 = Thread1()
	tab4 = Tab(th4)
	tab4.setupUi(CreateDataset_main_window,ui.tabWidget,ui.centralwidget)
	th5 = Thread1()
	tab5 = Tab(th5)
	tab5.setupUi(CreateDataset_main_window,ui.tabWidget,ui.centralwidget)
	th6 = Thread1()
	tab6 = Tab(th6)
	tab6.setupUi(CreateDataset_main_window,ui.tabWidget,ui.centralwidget)
	th7 = Thread1()
	tab7 = Tab(th7)
	tab7.setupUi(CreateDataset_main_window,ui.tabWidget,ui.centralwidget)
	th8 = Thread1()
	tab8 = Tab(th8)
	tab8.setupUi(CreateDataset_main_window,ui.tabWidget,ui.centralwidget)
	th9 = Thread1()
	tab9 = Tab(th9)
	tab9.setupUi(CreateDataset_main_window,ui.tabWidget,ui.centralwidget)
	th10 = Thread1()
	tab10 = Tab(th10)
	tab10.setupUi(CreateDataset_main_window,ui.tabWidget,ui.centralwidget)
	CreateDataset_main_window.show()
	sys.exit(app.exec_())