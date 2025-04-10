import tkinter as tk
from tkinter import ttk, messagebox
from keras.models import load_model
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

class DiabetesPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Diabetes Risk Prediction System")
        self.root.geometry("900x600")
        self.root.configure(bg='#f0f8ff')
        
        # Load model
        try:
            self.model = load_model('diabetes_risk_nn.h5')
        except:
            messagebox.showerror("Error", "Model file not found. Please train the model first.")
            return
        
        # Load feature descriptions
        self.feature_descriptions = {
            0: "Number of times pregnant",
            1: "Plasma glucose concentration (2 hours in oral glucose tolerance test)",
            2: "Diastolic blood pressure (mm Hg)",
            3: "Triceps skin fold thickness (mm)",
            4: "2-Hour serum insulin (mu U/ml)",
            5: "Body mass index (weight in kg/(height in m)^2)",
            6: "Diabetes pedigree function",
            7: "Age (years)"
        }
        
        self.create_widgets()
    
    def create_widgets(self):
        # Header
        header_frame = tk.Frame(self.root, bg='#4682b4')
        header_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(header_frame, text="Diabetes Risk Prediction", font=('Arial', 20, 'bold'), 
                bg='#4682b4', fg='white').pack(pady=10)
        
        # Input Frame
        input_frame = tk.LabelFrame(self.root, text="Patient Information", font=('Arial', 12), 
                                  bg='#f0f8ff', padx=10, pady=10)
        input_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.entries = []
        for i in range(8):
            row = i // 4
            col = i % 4
            
            frame = tk.Frame(input_frame, bg='#f0f8ff')
            frame.grid(row=row, column=col, padx=5, pady=5, sticky='w')
            
            tk.Label(frame, text=f"Feature {i+1}:", bg='#f0f8ff').pack(anchor='w')
            tk.Label(frame, text=self.feature_descriptions[i], font=('Arial', 8), 
                    bg='#f0f8ff', fg='#555').pack(anchor='w')
            
            entry = ttk.Entry(frame, width=15)
            entry.pack(pady=(0, 5))
            self.entries.append(entry)
        
        # Button Frame
        button_frame = tk.Frame(self.root, bg='#f0f8ff')
        button_frame.pack(pady=10)
        
        predict_btn = ttk.Button(button_frame, text="Predict Diabetes Risk", 
                               command=self.predict)
        predict_btn.pack(side='left', padx=5)
        
        clear_btn = ttk.Button(button_frame, text="Clear Fields", 
                             command=self.clear_fields)
        clear_btn.pack(side='left', padx=5)
        
        # Result Frame
        self.result_frame = tk.LabelFrame(self.root, text="Prediction Result", 
                                        font=('Arial', 12), bg='#f0f8ff')
        self.result_frame.pack(fill='x', padx=20, pady=10)
        
        self.result_label = tk.Label(self.result_frame, text="", font=('Arial', 14), 
                                   bg='#f0f8ff')
        self.result_label.pack(pady=10)
        
        # Risk Meter
        self.risk_meter = ttk.Progressbar(self.result_frame, orient='horizontal', 
                                        length=300, mode='determinate')
        self.risk_meter.pack(pady=10)
        
        self.risk_label = tk.Label(self.result_frame, text="Risk Level: 0%", 
                                  font=('Arial', 10), bg='#f0f8ff')
        self.risk_label.pack()
        
        # Footer
        footer_frame = tk.Frame(self.root, bg='#4682b4')
        footer_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(footer_frame, text="© 2025  Diabetes Prediction System", 
                bg='#4682b4', fg='white').pack()
    
    def predict(self):
        try:
            input_data = []
            for entry in self.entries:
                value = float(entry.get())
                input_data.append(value)
            
            input_array = np.array([input_data])
            prediction = self.model.predict(input_array)[0][0]
            risk_percentage = round(prediction * 100, 2)
            
            self.risk_meter['value'] = risk_percentage
            self.risk_label.config(text=f"Risk Level: {risk_percentage}%")
            
            if risk_percentage < 30:
                result_text = "Low Risk of Diabetes"
                color = "green"
            elif risk_percentage < 70:
                result_text = "Moderate Risk of Diabetes"
                color = "orange"
            else:
                result_text = "High Risk of Diabetes"
                color = "red"
            
            self.result_label.config(text=result_text, fg=color)
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numerical values for all fields")
    
    def clear_fields(self):
        for entry in self.entries:
            entry.delete(0, 'end')
        
        self.risk_meter['value'] = 0
        self.risk_label.config(text="Risk Level: 0%")
        self.result_label.config(text="")

if __name__ == "__main__":
    root = tk.Tk()
    app = DiabetesPredictorApp(root)
    root.mainloop()