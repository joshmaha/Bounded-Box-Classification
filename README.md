![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-cv2-green?logo=opencv)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-blue?logo=numpy)
![Course](https://img.shields.io/badge/Course-CAI4841%20Computer%20Vision-red)
![University](https://img.shields.io/badge/University-University%20of%20South%20Florida-darkgreen)

<p align="center">
<img width="405" height="644" alt="image" src="https://github.com/user-attachments/assets/0dcd40ed-9729-4094-8a4b-c653852521ba" />
</p>
# Bounded Box Classification 

This project focuses on **image-based object classification using bounding boxes**, where regions of interest are localized and classified using computer vision techniques. The goal is to accurately identify objects within images by first constraining the search space using bounding boxes, then applying classification logic to those regions.

This repository serves as an experimental and educational project exploring object localization, feature extraction, and classification pipelines.

---

## Project Overview

In many real-world computer vision applications (e.g., robotics, autonomous vehicles, surveillance, and medical imaging), it is inefficient or impractical to classify an entire image at once. Instead, **bounded box classification** allows systems to:

- Focus computation on relevant regions  
- Improve classification accuracy  
- Reduce noise and background interference  

This project demonstrates how bounding boxes can be used as a preprocessing step for object classification.

---

## Key Concepts Covered

- Bounding box representation (x, y, width, height)
- Region of Interest (ROI) extraction
- Image preprocessing and normalization
- Feature extraction from bounded regions
- Supervised classification of detected objects
- Evaluation of classification performance

---


## Installation & Dependencies

This project is written in Python 3 and works with basic imaging libraries.

1. Clone the repository:
   ```bash
   git clone https://github.com/joshmaha/Bounded-Box-Classification.git
   cd Bounded-Box-Classification

2. Create the environment
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   venv\Scripts\activate      # Windows PowerShell

  3. Install Dependencies
     ```bash
     pip install -r requirements.txt
   
   4. Visualize Annotations
      ```bash
      $ python draw.py -f /home/user/1.png -a /home/user/1.txt


---

##  How It Works
### Bounding Box Concept
A bounding box is a rectangle defined over an image region that encloses an object of interest. The box can be represented with either:
- Coordinates of two corners (x1, y1, x2, y2), or
- A center point with a width/height (x, y, w, h). 

The scripts in this project use Python functions to:
- Compute bounding boxes around shapes or features.
- Calculate metrics like Intersection Over Union (IOU) to evaluate overlap.
- Draw these boxes onto images for visualization.

Take a look at this example test cases and the associated coordinated for Rocky
<p align="center">
<img width="1338" height="713" alt="image" src="https://github.com/user-attachments/assets/6276f937-f38d-49c0-a031-a3ef1ae88728" align="center"/>
</p>

--- 
## Acknowledgements



