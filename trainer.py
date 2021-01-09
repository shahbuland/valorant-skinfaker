from PyQt5.QtWidgets import QApplication, QLabel, QWidget,QFileDialog, QPushButton, QInputDialog, QCheckBox, QMessageBox, QLineEdit, QHBoxLayout
import os
import sys
import numpy as np

from extractGen import extract_from_path
from CycleGAN.models import CycleGAN
from CycleGAN.train import train
from vids2data.vid_io import vid2arr

CHECKPOINT_PATH = None
EXPERIMENT_NAME = None
INTERVALS = [None, None, None] # Log, sample and checkpoint intervals
DATA_A_PATH = None
DATA_B_PATH = None


# Checks if experiment directory exists and possibly creates it
# Returns true if the directory is available, false otherwise
def set_experiment(name):
    if not os.path.exists("experiments/" + name):
        os.mkdir("experiments/" + name)
    
    for subdir in ["source_A", "source_B", "samples", "checkpoints"]:
        if not os.path.exists("experiments/" + name + "/" + subdir):
            os.mkdir("experiments/" + name + "/" + subdir)
            
    if not os.path.exists("experiments/" + name + "/prefs.txt"):
        prefs = open("experiments/" + name + "/prefs.txt", "x")
        prefs.write("LOG 100\nSAMPLE 100\nCHECKPOINT 500")
        prefs.close()
    
    with open("experiments/" + name + "/prefs.txt", "r") as prefs:
        for i, line in enumerate(prefs):
            if i > 2: break
            
            _, val = line.split()
            INTERVALS[i] = int(val)
    
    global EXPERIMENT_NAME
    EXPERIMENT_NAME = name
    
    return True
    
def save_prefs():
    if EXPERIMENT_NAME is None:
        print("ERROR: can't save prefs; no experiment selected")
        return
    for val in INTERVALS:
        if val is None:
            print("ERROR: some interval field is empty; can't save")
            return
    with open("experiments/" + EXPERIMENT_NAME + "/prefs.txt", "w") as prefs:
        prefs.write("LOG ")
        prefs.write(str(INTERVALS[0]))
        prefs.write("\nSAMPLE ")
        prefs.write(str(INTERVALS[1]))
        prefs.write("\nCHECKPOINT ")
        prefs.write(str(INTERVALS[2]))
        prefs.close()
        
def begin_training():
    model = CycleGAN()
    if CHECKPOINT_PATH is not None:
        model.load_checkpoint_from_path(CHECKPOINT_PATH)
    model.cuda()
    A = np.load("experiments/" + EXPERIMENT_NAME + "/" + "A_data.npy")
    B = np.load("experiments/" + EXPERIMENT_NAME + "/" + "B_data.npy")
    
    # Train on A and B for 200 epochs
    train(model, A, B, 200, INTERVALS[0], INTERVALS[1], INTERVALS[2], "experiments/" + EXPERIMENT_NAME + "/")
    
def convert_src(frame_skip):
    try:
        frame_skip = int(frame_skip)
    except:
        print("ERROR: Frame skip value wasn't valid")
        return 
        
    if DATA_A_PATH is None or DATA_B_PATH is None:
        print("ERROR: Can't find any source videos (Folder not found)")
        return
        
    A_vids = [DATA_A_PATH + "/" + path for path in os.listdir(DATA_A_PATH)]
    B_vids = [DATA_B_PATH + "/" + path for path in os.listdir(DATA_B_PATH)]
    
    if not A_vids or not B_vids:
        print("ERROR: Can't find source videos (One or both folders empty)")
        
    A_res = "experiments/" + EXPERIMENT_NAME + "/" + "A_data.npy"
    B_res = "experiments/" + EXPERIMENT_NAME + "/" + "B_data.npy"
    
    vid2arr(A_vids, A_res, frame_skip)
    vid2arr(B_vids, B_res, frame_skip)

class Startup(QWidget):
    def __init__(self):
        super(Startup, self).__init__()
        self.setWindowTitle("Skinfake Trainer")
        self.setGeometry(100, 100, 400, 150)
        
        self.NEXT = None
        
        self.exp_label = QLabel("Experiment Name:", self)
        self.exp_label.move(5,10)
        self.exp = QLineEdit(self)
        self.exp.move(5, 40)
        self.exp_confirm = QPushButton("Ok", self)
        self.exp_confirm.move(5, 80)
        self.exp_confirm.clicked.connect(self.load_experiment)
        
        self.show()
    
    def load_experiment(self):
        if self.exp.text() == "":
            return
        else:
            name = self.exp.text()
            if not os.path.exists("experiments/" + name):
                qm = QMessageBox
                ans = qm.question(self, '', "Can't find experiment. Create experiment titled " + name + "?", qm.Yes | qm.No)
                if ans == qm.No:
                    return
        
        set_experiment(name)
        self.hide()
        
        global CHECKPOINT_PATH
        global DATA_A_PATH
        global DATA_B_PATH
        CHECKPOINT_PATH = None
        DATA_A_PATH = "experiments/" + name + "/source_A"
        DATA_B_PATH = "experiments/" + name + "/source_B"
        
        self.NEXT.setPrompts()
        self.NEXT.show()
    
    def set_next(self, n):
        self.NEXT = n

class TrainingScreen(QWidget):
    def __init__(self):
        super(TrainingScreen, self).__init__()
        self.setWindowTitle("Skinfake Trainer")
        self.setGeometry(100, 100, 800, 400)
        
        self.NEXT = None
        
        self.close = QPushButton("Close", self)
        self.close.move(5, 10)
        self.close.clicked.connect(self.to_startup)
        
        self.load_cp = QPushButton("Load Checkpoint", self)
        self.load_cp.move(5, 50)
        self.load_cp.clicked.connect(self.open_checkpoint)
        
        self.save_interval_vals = QPushButton("Save Intervals", self)
        self.save_interval_vals.move(190, 90)
        self.save_interval_vals.clicked.connect(self.save_intervals)
        
        self.log_interval_label = QLabel("Log Interval: ", self)
        self.log_interval = QLineEdit(self)
        self.log_interval_label.move(5, 75)
        self.log_interval.move(30, 90)
        
        self.sample_interval_label = QLabel("Sample Interval: ", self)
        self.sample_interval = QLineEdit(self)
        self.sample_interval_label.move(5, 115)
        self.sample_interval.move(30, 130)
        
        self.checkpoint_interval_label = QLabel("Checkpoint Interval: ", self)
        self.checkpoint_interval = QLineEdit(self)
        self.checkpoint_interval_label.move(5, 155)
        self.checkpoint_interval.move(30, 170)
        
        self.extract = QPushButton("Extract Generator", self)
        self.extract.move(5, 210)
        self.extract.clicked.connect(self.try_extract)
        
        self.convert = QPushButton("Convert Videos", self)
        self.convert.move(5, 250)
        self.convert.clicked.connect(self.try_convert)
        
        self.conv_frame_skip_label = QLabel("Frame Skip: ", self)
        self.conv_frame_skip = QLineEdit(self)
        self.conv_frame_skip_label.move(100, 235)
        self.conv_frame_skip.move(100, 250)
        
        
        self.start_train = QPushButton("Train", self)
        self.start_train.move(5, 330)
        self.start_train.clicked.connect(begin_training)
    
    def save_intervals(self):
        try:
            INTERVALS[0] = int(self.log_interval.text())
            INTERVALS[1] = int(self.sample_interval.text())
            INTERVALS[2] = int(self.checkpoint_interval.text())
        except:
            print("ERROR: One or more intervals aren't integer values")
            return
        
        save_prefs()
            
    def set_next(self, n):
        self.NEXT = n
    
    def to_startup(self):
        self.hide()
        self.NEXT.show()
    
    def open_checkpoint(self):
        global CHECKPOINT_PATH
        path, _ = QFileDialog.getOpenFileName(caption = 'Open file', directory = 'experiments/' + EXPERIMENT_NAME + '/checkpoints', filter = "Parameters (*.pt);; All files (*.*)")
        CHECKPOINT_PATH = path
    
    # Sets all prompts to current runtime values
    def setPrompts(self):
        self.log_interval.setText(str(INTERVALS[0]))
        self.sample_interval.setText(str(INTERVALS[1]))
        self.checkpoint_interval.setText(str(INTERVALS[2]))
    
    def try_extract(self):
        if CHECKPOINT_PATH is None:
            print("ERROR: Trying to extract generator when no model is loaded")
        elif EXPERIMENT_NAME is None:
            print("ERROR: No experiment loaded")
        else:
            extract_from_path(CHECKPOINT_PATH, "experiments/" + EXPERIMENT_NAME + "/")
    
    def try_convert(self):
        convert_src(self.conv_frame_skip.text())
        
if __name__ == "__main__":
    # Establish experiments folder
    if not os.path.exists("experiments"):
        os.mkdir("experiments")
    app = QApplication(sys.argv)
    ex1 = Startup()
    ex2 = TrainingScreen()
    ex1.set_next(ex2)
    ex2.set_next(ex1)
    sys.exit(app.exec_())