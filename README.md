# **Fashion-MNIST Classification with Custom MLP and CNN Models**

This project demonstrates image classification on the Fashion-MNIST dataset using a custom-implemented Multilayer Perceptron (MLP) and a Convolutional Neural Network (CNN) within a Jupyter Notebook (`EE954_CNN_MLP_Group10.ipynb`). The MLP is built from scratch, including manual forward and backward passes, while the CNN leverages PyTorch's built-in modules. The notebook integrates both models to explore their performance and provides insights into neural network training and evaluation.

---

## **Table of Contents**

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Notebook](#running-the-notebook)
- [Notebook Structure](#notebook-structure)
- [Results](#results)
- [Additional Notes](#additional-notes)
- [License](#license)
- [Contact](#contact)

---

## **Prerequisites**

Before running the notebook, ensure that you have the following installed:

- **Python 3.6+**
- **Jupyter Notebook or JupyterLab**
- **Required Python Libraries:**
  - `torch`
  - `torchvision`
  - `numpy`
  - `matplotlib`
  - `seaborn`
---

## **Installation**

### **1. Clone the Repository**

If you haven't already, clone the repository to your local machine:

```bash
git clone https://github.com/neeleshbatham/mlp-cnn-fashionmnist.git
cd mlp-cnn-fashionmnist
```


### **2. Set Up a Virtual Environment (Recommended)**
Creating a virtual environment helps manage dependencies and prevents conflicts with other projects.

#### Using venv:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Using conda:
```bash
conda create -n fashion_mnist_env python=3.8
conda activate fashion_mnist_env
```

### **3. Install Required Libraries**
Install the necessary Python packages:

```bash
pip install torch torchvision numpy matplotlib seaborn
```
Alternatively, if a requirements.txt file is provided:
```bash
pip install -r requirements.txt
```

## **Running the Notebook**

1. Launch Jupyter Notebook
From the project directory, start Jupyter Notebook:

```bash
jupyter notebook
```

Alternatively, if you prefer JupyterLab:
```bash
jupyter lab
```
This will open the Jupyter interface in your default web browser.

## 2. Open the Notebook
In the Jupyter interface, navigate to and open the notebook file:

```bash
EE954_CNN_MLP_Group10.ipynb
```

## 3. Run the Notebook Cells
**``Option 1``**: Run All Cells
In the notebook menu, select Kernel > Restart & Run All.
This will execute all cells sequentially from the beginning.

**``Option 2``**: Run Cells Individually
Start by running the first cell to import necessary libraries.
Proceed to run each cell in order by clicking Run (▶️) or pressing Shift + Enter.
This approach allows you to observe outputs and understand each step.

## 4. Monitor Training Progress
Training Output: The notebook will display training progress for both MLP and CNN models, including loss and accuracy per epoch.
Training Time: Training times for both models are printed after their respective training loops.
Evaluation Metrics: The notebook evaluates both models on the test dataset and prints metrics such as test loss and accuracy.

## 5. Visualize Results (Optional)
If the notebook includes visualization cells (e.g., plotting loss curves), ensure you run these cells to see the graphs.
Plots may require matplotlib; ensure it's installed.


#  **Results** 
MLP Model:

- Training Accuracy: ~85%
- Validation Accuracy: ~84%
- Test Accuracy: ~84%
- Training Time: Varies based on hardware (e.g., ~2 minutes on CPU)

CNN Model:
- Training Accuracy: ~93%
- Validation Accuracy: ~91%
- Test Accuracy: ~91%
- Training Time: Varies based on hardware (e.g., ~50 minutes on CPU)

Note: Actual results and training times may vary depending on your system configuration and any modifications to the code or hyperparameters.

Additional Notes
Hardware Acceleration:

If you have a CUDA-compatible GPU, you can modify the code to utilize it for faster training.
Add device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') and move models and tensors to the device using .to(device).



## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For any questions, suggestions, or feedback, please contact 
- Neelesh Batham: neeleshb24@iitk.ac.in 
- Karan Arora: karanarora23@iitk.ac.in 
- Advait Patwardhan: advaitmp24@iitk.ac.in
- Tony Varghese: tonyv23@iitk.ac.in