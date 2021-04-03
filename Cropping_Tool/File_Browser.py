"""
Created on Wed Mar, 3, 2021
@author: Nhu Nhat Anh
"""

import tkinter
from tkinter import filedialog
from tkinter import ttk
import os

"""
Allows user to select an image in the system and Returns the directory
"""
def file_browser():
    root = tkinter.Tk()
    root.withdraw()
    file_dir = filedialog.askopenfile(mode = 'r', parent = root, initialdir = "./", title='Choose an Image', filetypes = (("ALL FILE", "*.*"), ("JPG", "*.jpg"), ("PNG", "*.png"), ("DICOM", "*.dicom")))
    
    return file_dir.name # directory to the file