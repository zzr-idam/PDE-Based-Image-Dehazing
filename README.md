# Image Dehazing: A Deep - Learning - Based Approach

## 1.        Project Overview
This GitHub repository hosts an image dehazing project powered by deep - learning techniques.        Our goal is to effectively remove haze from input images, restoring clear and visually appealing results.        The core dehazing logic is encapsulated within the provided codebase, enabling both research experimentation and practical application for dehazing tasks.

## 2.        Related Paper
The dehazing algorithm implemented in this project is based on the research presented in:
**Paper Title**: *A PDE-Based Image Dehazing Method via Atmospheric Scattering Theory*
**Authors**: *Zhuoran Zheng*
**DOI/Link**: *https://arxiv.org/pdf/2506.08793*

This project serves as an implementation and extension of the ideas and methodologies put forward in the paper, facilitating the reproduction and further exploration of image dehazing using deep - learning.

## 3.        Repository Structure
```
.
├── dehazing_samples/   # Directory to store test hazy images and related samples
├── demo.py             # Main script for dehazing execution.        Contains the pipeline from image input to dehazed output.
└── README.       md           # The document you are currently reading, providing project details and usage instructions.
```

## 4.        Dependencies
To run this deep - learning - based image dehazing project, the following Python libraries are required:
- **PyTorch**: Serves as the foundation for building and training the deep - learning models (if applicable) and performing tensor - related operations crucial for the dehazing algorithm.
- **OpenCV - Python (`opencv - python`)**: Used for image I/O operations (reading hazy images and writing dehazed results) as well as essential image processing tasks during the dehazing pipeline.
- **NumPy**: Provides support for efficient numerical computations, which are fundamental for manipulating image data in the form of arrays.

You can install these dependencies using either `pip` or `conda` (recommended for managing deep - learning environments).        For `pip`, use the following commands:
```bash
pip install torch opencv - python numpy
```
If you are using `conda`, the commands would be:
```bash
conda install pytorch opencv numpy -c pytorch
```

## 5.        How to Run
### Step 1: Prepare Input Images
Place the hazy images that you want to dehaze inside the `dehazing_samples/` directory.        Make sure the images are in a supported format (e.g., `.jpg`, `.png`).

### Step 2: Execute the Dehazing Script
Run the `demo.py` script using the following command in your terminal (assuming you have navigated to the project's root directory):
```bash
python demo.py
```
This script will:
1.        Read the hazy images from `dehazing_samples/`.
2.        Apply the deep - learning - based dehazing algorithm.
3.        Output the dehazed images (the output path can be customized within the `demo.py` script).


## 6.        Notes and Tips
- **Environment Compatibility**: Ensure that your Python environment (especially the versions of PyTorch, CUDA, and cuDNN if using GPU acceleration) is compatible with the requirements of the deep - learning operations in the project.        In case of GPU - related issues, refer to the official documentation of PyTorch for troubleshooting.
- **Input Image Requirements**: The input hazy images should be of reasonable size.        Extremely large images may cause memory issues during processing.        You can consider resizing the images beforehand or adjusting the memory - related settings in the code if needed.
- **Customization**: The `demo.py` script can be further customized.        For example, you can modify the dehazing parameters, adjust the output image quality settings, or integrate additional post - processing steps for better visual results.
