# Model Card for Facial Expression Recognition Model

This model card provides an overview of a Convolutional Neural Network (CNN) developed for facial expression recognition. The project aimed to explore the effectiveness of various strategies in handling unbalanced datasets, particularly focusing on the impact of the `CategoricalFocalCrossentropy()` loss function and adjustments in the model's architecture and hyperparameters. The model was developed and tested using Python, TensorFlow, and Pandas within Google Colab, leveraging GPU acceleration for enhanced processing speeds.

## Model Details

### Model Description

The CNN model was trained on a dataset reduced by 10% of the original size to facilitate faster training speeds in Google Colab. Despite the reduction, the dataset maintained the original distribution of data across all classes of facial expressions. The training and testing splits were directly managed from Google Colab's content folder, with the data zip folder required to be uploaded to Google Colab during runtime.

- **Developed by:** Joao Pedro dos Santos, with critiques from Joshua Brewington and Johnny Duenas.
- **Model type:** Convolutional Neural Network (CNN) for facial expression recognition.
- **Language(s):** Python
- **Libraries/Frameworks:** TensorFlow, Pandas
- **License:** Open Source

### Model Sources

- **Repository:** [GitHub Repository](https://github.com)
- **Paper [optional]:** [Facial Expression Recognition with TensorFlow](https://blog.devgenius.io/facial-expression-recognition-with-tensorflow-90f6174163c3)
- **Additional Sources:**
  - [L1 vs L2 Regularization in Machine Learning](https://towardsdatascience.com/l1-vs-l2-regularization-in-machine-learning-differences-advantages-and-how-to-apply-them-in-72eb12f102b5)
  - [Focal Loss: What, Why, and How](https://medium.com/swlh/focal-loss-what-why-and-how-df6735f26616)

## Uses

### Direct Use

This model is designed for the direct recognition of facial expressions from images, suitable for applications requiring emotional analysis, such as customer feedback systems, psychological research, and interactive entertainment technologies.

### Downstream Use [optional]

The model can be fine-tuned for specific tasks within the domain of facial expression recognition, adapting to detect subtle emotional cues or focusing on a particular demographic.

### Out-of-Scope Use

The model is not intended for identifying individuals, predicting personal information, or any form of surveillance.

## Bias, Risks, and Limitations

Despite efforts to achieve higher accuracies, the model's performance may vary when testing different classes.. The initial layer's neurons were found to be oversaturated when all 7 classes were trained, indicating a potential limitation in the model's architecture for handling complex, unbalanced datasets.

### Recommendations

Users should consider these limitations and potentially validate the model further in critical applications. Continuous research and development are recommended to enhance the model's robustness and inclusivity.

## How to Get Started with the Model

Refer to the [Facial Expression Recognition with TensorFlow](https://blog.devgenius.io/facial-expression-recognition-with-tensorflow-90f6174163c3) blog post for detailed implementation instructions, including code snippets and data preprocessing guidelines.

## Training Details

### Training Data

The model was trained on a dataset reduced to 10% of the FER-2013 dataset size, ensuring the same distribution of emotions to address class imbalance. The data was unploaded to co-lab's runtime in its contents folder.

### Training Procedure

#### Preprocessing

Images were resized to 48x48 pixels and normalized. Data augmentation techniques such as rotation and zoom were applied to enhance the diversity of the training data. This was done by the use of tensorflow's import 'ImageDataGenerator'.

#### Training Hyperparameters

- **Training regime:** Utilized the `CategoricalFocalCrossentropy()` loss function to focus on hard-to-classify examples and mitigate the impact of class imbalance. While the loss function did not improve accuracy, it significantly reduced the loss.

## Evaluation

### Testing Data, Factors & Metrics

The model was evaluated on a separate test set, with experiments conducted on different models with fewer classes (6 and 4), which demonstrated high accuracies.

### Results

The use of `CategoricalFocalCrossentropy()` and GPU acceleration in Google Colab facilitated faster processing speeds and a significant reduction in loss, despite the challenges posed by the unbalanced dataset.

## Technical Specifications

Train and test datasets were ran from the google co-lab's content folder to achieve a faster runtime.

### Model Architecture and Objective

The CNN architecture was optimized for feature extraction and classification of facial expressions, with a focus on achieving high accuracy across all classes, despite the unbalanced nature of the training data.

### Compute Infrastructure

Training leveraged Google Colab's GPU acceleration, enabling faster processing speeds and efficient handling of the computational demands of the CNN architecture.

## Citation

**APA:**

dos Santos, J. P., Brewington, J., & Duenas, J. (2023). Facial Expression Recognition with TensorFlow. *DevGenius*. Retrieved from https://blog.devgenius.io/facial-expression-recognition-with-tensorflow-90f6174163c3

**BibTeX:**

```bibtex
@article{facialexpressionrecognition2023,
  title={Facial Expression Recognition with TensorFlow},
  author={dos Santos, Joao Pedro and Brewington, Joshua and Duenas, Johnny},
  journal={DevGenius},
  year={2023},
  url={https://blog.devgenius.io/facial-expression-recognition-with-tensorflow-90f6174163c3}
}
```

## More Information

For further details and updates, please refer to the [GitHub Repository](https://github.com) and the [Facial Expression Recognition with TensorFlow](https://blog.devgenius.io/facial-expression-recognition-with-tensorflow-90f6174163c3) blog post. Additional insights into the model's development and performance can be found in the articles on L1 vs L2 Regularization and Focal Loss.
