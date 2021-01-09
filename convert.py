from PyQt5.QtWidgets import QApplication, QLabel, QWidget,QFileDialog, QPushButton, QInputDialog, QCheckBox
from vidConverter import convert
import sys

SRC_PATH = None
MODEL_PATH = None
STITCH = False
MASK = False
HALF = False
CUDA = False

# Opens file dialog to let user update src
def update_src():
    global SRC_PATH
    path, _ = QFileDialog.getOpenFileName(caption = 'Open file', directory = './', filter = "MP4 (*.mp4);; AVI (*.avi);; All files (*.*)")
    SRC_PATH = path
    
# Opens file dialog to let user update model
def update_model():
    global MODEL_PATH
    path, _ = QFileDialog.getOpenFileName(caption = 'Open file', directory = './models/', filter = "Parameters (*.pt);; All files (*.*)")
    MODEL_PATH = path

# Calls convert script 
def call_convert():
    if SRC_PATH is None or MODEL_PATH is None: return
    
    convert(SRC_PATH, SRC_PATH[:-4] + "_converted.avi", MODEL_PATH, STITCH, MASK, CUDA, HALF)
   # except Exception as e:
    #    print("Error during generation:")
     #   print(e)
        
    
def update_stitch():
    global STITCH
    STITCH = not STITCH

def update_mask():
    global MASK
    MASK = not MASK

def update_half():
    global HALF
    HALF = not HALF
    
def update_cuda():
    global CUDA
    CUDA = not CUDA
    
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("Skinfake Converter")
    window.setGeometry(100, 100, 400, 150)
    window.move(60, 15)
    
    srcFileButton = QPushButton("Open Source Video", window)
    srcFileButton.move(5, 15)
    srcFileButton.clicked.connect(update_src)
    
    modelFileButton = QPushButton("Open Model", window)
    modelFileButton.move(5, 15 + 30)
    modelFileButton.clicked.connect(update_model)
    
    generateButton = QPushButton('Generate!', window)
    generateButton.move(5,15+60)
    generateButton.clicked.connect(call_convert)

    stitchButton = QCheckBox('Stitch', window)
    stitchButton.move(5+100,15)
    stitchButton.toggled.connect(update_stitch)
                      
    maskButton = QCheckBox('Mask', window)
    maskButton.move(5+100,15+30)
    maskButton.toggled.connect(update_mask)
                      
    halfButton = QCheckBox('Half Precision', window)
    halfButton.move(5+100,15+60)
    halfButton.toggled.connect(update_half)
    
    cudaButton = QCheckBox('CUDA (Use GPU)', window)
    cudaButton.move(5+100,15+90)
    cudaButton.toggled.connect(update_cuda)
                      
    window.show()
    sys.exit(app.exec_())
