# **mllm-physical-design**

## **IR Drop Prediction**

This project provides tools for pairwise evaluation of IR Drop severity using new features. Ensure that you add your OpenAI API key in the `inference.py` file before running any scripts.

---

### **Features**

- **RGB Grid:** Visualizes high-severity regions with color-coded intensity ranges for better understanding.
- **Centroid and Cluster Analysis:** Calculates centroids of high-severity clusters and analyzes spatial relationships, such as inter-cluster distances.
- **Severity Score Calculation:** Computes IR Drop severity based on features like cluster density and total red area, providing a quantitative measure for severity evaluation.

---

### **Required Packages**

The following Python packages are required:
- `numpy`
- `scipy`
- `Pillow` (PIL)
- `argparse`

Install the dependencies using:
```bash
pip install numpy scipy Pillow
```

---

### **Example Usage**

Run the `inference.py` script for pairwise evaluation of IR Drop severity using features like grid or distance metrics:

```bash
python inference.py --num_pairs 3 \
                    --feature_path "/data2/NVIDIA/CircuitNet-N28/Dataset/IR_drop/feature" \
                    --label_path "/data2/NVIDIA/CircuitNet-N28/Dataset/IR_drop/label" \
                    --with_grid
```

**Command-Line Arguments:**
- `--num_pairs`: Number of IR Drop map pairs to evaluate (e.g., 3).
- `--feature_path`: Path to the folder containing feature files.
- `--label_path`: Path to the folder containing label files.
- `--with_grid`: Optional flag to include grid features in the evaluation.

---

### **How It Works**
1. **Setup API Key:**
   - Add your OpenAI API key in the `inference.py` file under the appropriate section.

2. **Run Pairwise Evaluation:**
   - Use the above example to perform pairwise evaluation of severity features between IR Drop map pairs.

3. **View Results:**
   - The results will be printed in the terminal or logged for further analysis. 

---
