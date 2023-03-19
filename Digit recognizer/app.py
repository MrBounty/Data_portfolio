from tkinter import *
from tkinter import filedialog, ttk
from PIL import ImageGrab
from numpy import array
from AI import *
import pathlib


class Menu(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'
    CANVAS_SIZE = 200

    def __init__(self):
        #Init some bool
        self.train_file_selected = False
        self.test_file_selected = False
        self.AI_created = False

        #Init the window and change some stuff
        self.root = Tk()
        self.root.title("Digit reconizer AI with Keras")
        self.root.iconbitmap('icon.ico')

        actual_row = 0

        #Make AI GUI part
        actual_row = self.makeAI_UI(actual_row)

        actual_row += 1

        #make the paint GUI part
        actual_row = self.makeCanvas(actual_row)

        #Finish setup and start the GUI
        self.setup()
        self.root.after(200, self.predict_number)
        self.root.mainloop()

    def setup(self):
        #Init some variable and add the paint function to the canvas
        self.old_x = None
        self.old_y = None
        self.line_width = 5
        self.color = self.DEFAULT_COLOR
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def makeCanvas(self, actual_row):
        #this is the GUI part to draw and predict

        #Make the 2 button clean and predict
        self.clean_button = Button(self.root, text='clean', command=self.clean)
        self.clean_button.grid(row=actual_row, column=0, sticky='nesw', columnspan = 2)

        #self.predict_button = Button(self.root, text='Predict', command=self.predict_number, state='disabled')
        #self.predict_button.grid(row=actual_row, column=1, sticky='nesw')

        actual_row += 1

        #Make the canvas
        self.c = Canvas(self.root, bg='white', width=self.CANVAS_SIZE, height=self.CANVAS_SIZE)
        self.c.grid(row=actual_row, columnspan=5)

        actual_row += 1

        #Make the label where we output the predicted number
        self.result_text = Label(self.root, text="Result: ")
        self.result_text.grid(row=actual_row, column=0, columnspan = 2)

        return actual_row

    def makeAI_UI(self, actual_row):
        #Make the first part of the GUI, dedicated for the AI

        #Make the 2 import train and test button
        self.open_train_button = Button(self.root, text='Open the train file', command=self.select_train_file)
        self.open_train_button.grid(row=actual_row, column=0, sticky='nesw')

        self.open_test_button = Button(self.root, text='Open the test file', command=self.select_test_file)
        self.open_test_button.grid(row=actual_row, column=1, sticky='nesw')

        actual_row += 1

        #Make the part to ask the number of epoch
        Label(self.root, text="Number of epoch").grid(row=actual_row, column=0)
        self.epoch_entry = Entry(self.root)
        self.epoch_entry.grid(row=actual_row, column=1)
        self.epoch_entry.insert(0, '1')

        actual_row += 1

        #Make the part to ask the batches divider value
        Label(self.root, text="Batch divider").grid(row=actual_row, column=0)
        self.batch_entry = Entry(self.root)
        self.batch_entry.grid(row=actual_row, column=1)
        self.batch_entry.insert(0, '32')

        actual_row += 1

        #Make the start training button
        self.lunch_training_button = Button(self.root, text='Start training AI', command=self.lunch_training, state='disabled')
        self.lunch_training_button.grid(row=actual_row, column=0, columnspan = 2, sticky='nesw')

        actual_row += 1

        #Make button to load and save model
        self.save_model_button = Button(self.root, text='Save model', command=self.save_model, state='disabled')
        self.save_model_button.grid(row=actual_row, column=0, sticky='nesw')

        self.load_model_button = Button(self.root, text='Load model', command=self.load_model)
        self.load_model_button.grid(row=actual_row, column=1, sticky='nesw')

        actual_row += 1

        l0 = Label(self.root, text='     ', bg='grey')
        l0.grid(column=0, row=actual_row, sticky='nesw', columnspan=2)

        return actual_row

    def select_train_file(self):
        #Ask for the path
        self.train_file_path = filedialog.askopenfilename(
            title='Open a file',
            initialdir=pathlib.Path(__file__).parent.resolve(),
            filetypes=(('CSV file', '*.csv'), ('All files', '*.*')))

        #make the button green
        self.open_train_button.configure(bg = "#90EE90")
        self.train_file_selected = True

        #If the test file is already load, make the train button usable
        if self.test_file_selected:
            self.lunch_training_button.config(state='normal')

    def select_test_file(self):
        #Ask for the file path
        self.test_file_path = filedialog.askopenfilename(
            title='Open a file',
            initialdir=pathlib.Path(__file__).parent.resolve(),
            filetypes=(('CSV file', '*.csv'), ('All files', '*.*')))

        #Make the button green
        self.open_test_button.configure(bg = "#90EE90")
        self.test_file_selected = True

        #If the train is already load, make the train button usable
        if self.train_file_selected:
            self.lunch_training_button.config(state='normal')

    def save_model(self):
        #Save the model to an file that contain the date and time
        self.AI.save_model()

    def load_model(self):
        #Create the AI class
        self.AI = AI(self, 0, 0)
        self.AI_created = True

        #Ask for the path of the saved model
        model_path = filedialog.askopenfilename(
            title='Open a file',
            initialdir=pathlib.Path(__file__).parent.resolve(),
            filetypes=(('Saved model', '*.h5'), ('All files', '*.*')))

        #Load the model to the AI, make the load model button green and make the predict button usable
        self.AI.load_model(model_path)
        self.load_model_button.configure(bg = "#90EE90")
        self.predict_button.config(state='normal')

    def lunch_training(self):
        #Create the AI class, import the data and make it train
        self.AI = AI(self, int(self.epoch_entry.get()), int(self.batch_entry.get()))
        self.AI_created = True
        self.AI.importData(self.train_file_path, self.test_file_path)
        self.AI.start_training()

        #make the save model button usable
        self.save_model_button.config(state='normal')

    def clean(self):
        #Clean the all canvas
        self.c.delete("all")

    def predict_number(self):
        self.root.after(200, self.predict_number)

        if self.AI_created and self.AI.can_predict:
            #Get canvas coord on screen
            x = self.c.winfo_rootx()
            y = self.c.winfo_rooty()

            #Get screenshoot of screen, crop it, resize it and make it grayscale
            im = ImageGrab.grab().crop((x, y, x + self.CANVAS_SIZE, y + self.CANVAS_SIZE))
            im = im.resize((28, 28)).convert('L')

            #Transform image to numpy array and make it from 0-1 instead of 0-255, reshape it to fit the model
            image_array = array(im)
            image_array = image_array.astype(float)
            image_array = - (image_array/255 - 1)
            image_array = image_array.reshape((1, 28, 28, 1))

            #Predit the number with the image generated
            self.AI.predict(image_array)

    def paint(self, event):
        #The function to paint on the canvas, took on internet
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=self.color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None