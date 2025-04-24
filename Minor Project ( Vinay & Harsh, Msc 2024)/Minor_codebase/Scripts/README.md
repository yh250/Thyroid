# Thyroid Classification - Training Pipeline

This folder contains the training scripts for our **Thyroid Disease Classification** project (Minor Project, MSc 2024). The core training logic resides in `train.py`, but several other files must be executed or imported prior to running it for the training process to function correctly.

---

## 📁 Folder Structure

```bash
.
├── Models
│   ├── AlexNet      
│   │   ├── build_alexnet.py
│   │   └── torch_build_alexnet.py
│   ├── MobileNet
│   │   ├── build_mobilenet_v2.py
│   │   └── torch_build_mobilenet_v2.py
│   ├── Resnet
│   │   ├── build_resnet_18.py
│   │   └── torch_resnet_18.py
│   └── simple_cnn
│       ├── 3CNN
│       │   └── 3CNN.ipynb
│       └── build_simple_cnn.py
├── train.py 
└── utils
    ├── Data_loader.py
    ├── PyTorchModelTrainer.py
    ├── Save_resuts_2.py
    ├── gpu.py 
    ├── save_results.py
    └── test_results.py 
```
About Directories and Files
- ```train.py```: has the core logic for model training
- ```/utils```: Contains all auxiliary files needed to run train.py
    - ```Data_loader.py```: contains the loader function used to pass files to ```model.fit``` for training
    - ```save_results.py```: contains functions for saving metrics, generating plots, and logging results
    - ```Save_resuts_2.py```: alternate to ```save_results.py```
    - ```PyTorchModelTrainer.py```: alternate to ```train.py```, use for ```pytorch``` framework.
    - ```gpu.py```: to test tensorflow-gpu functionality, i.e., GPU availability for training.
    - ```test_results.py```: test file for ```save_results.py``` functionality.
- ```/Models```: Contains all the model architectures for both TensorFlow and PyTorch frameworks,
    All the TensorFlow architectures are named as ```build_alexnet.py``` without a prefix while PyTorch frameworks have a prefix ```torch_```
    before filename, for example ```torch_build_alexnet.py```. 
---

## 🚀 Getting Started

### 1. **Set Up Dependencies**

Make sure your environment has the required libraries. You can use a virtual environment and install the dependencies using:

```bash
pip install -r requirements.txt
```
If requirements.txt is not yet available, ensure you have installed:
torch, torchvision, numpy, matplotlib, seaborn
scikit-learn, etc.

--
### 2. Prepare Model Architectures
Before running train.py, ensure the model architectures are properly imported and registered. You can explore and modify model structures in the models/ directory.

For example, if you're using AlexNet, make sure to import it in train.py like so:

python
Copy
Edit
from models.alexnet import build_alexnet
model = build_alexnet()
Repeat for other models like resnet, mobilenet, etc.

3. Activate Utility Functions
Some utility functions (e.g., saving metrics, generating plots, logging results) are defined in save_results.py. Ensure these are called where necessary inside train.py or elsewhere.

You may need to run or import them before or within the training script:

python
Copy
Edit
from save_results import save_model_history, plot_metrics
4. Run Training
Once everything is set, run the training script:

bash
Copy
Edit
python train.py
Ensure that you modify train.py to specify:

Dataset paths

Model to train

Training hyperparameters (batch size, learning rate, epochs, etc.)

🛠️ Customization
You can modify train.py to experiment with different architectures by changing the import and instantiation line:

python
Copy
Edit
from models.mobilenet import build_mobilenet
model = build_mobilenet()
You can also customize:

Loss functions

Optimizers

Metrics tracking

Augmentations

📌 Notes
Ensure model files inside models/ define a consistent function like build_<modelname>() returning a torch.nn.Module model.

The project currently supports small medical datasets and uses basic to moderately complex architectures for evaluation.

Augmentation, class balancing (like SMOTE or GANs), and additional preprocessing should be handled separately.

