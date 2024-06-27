# Measuring Social Bias in Face Perception of a Vision-Language Model

## Abstract

Measuring algorithmic bias is the first step towards deploying AI systems responsibly. We introduce a new method to assess biases in the perception of human faces by vision-language models. Our novel method integrates three elements: Direct analysis of embeddings, the use of well-validated language terms from social psychology, and the use of a synthetic experimental dataset of faces. Our method measures social biases of face perception in CLIP, the most popular open-source embedding. 

The face images are varied systematically and independently by the legally protected attributes of age, gender, and race. The images also independently vary three other visual face attributes: smiling, lighting, and pose. Our results are experimental rather than observational (wild-collected). The dataset avoids any possible confounds in observational data that can occur between protected attributes and other image features that happen to correlate with those attributes.

Our analyses reveal that variations in protected attributes do systematically impact the model's social perception of faces. Unprotected attributes are varied independently but also impact social perceptions—for example, smiling impacts social perception as much as gender and more than age. Controlling for unprotected visual attributes is therefore necessary when assessing bias in protected attributes. Faces within each of the three demographic groups are rated similarly in psychological warmth and competence, but these perceptions differ across demographic groups. There is a strong pattern of bias that is special to what language is associated with the faces of Black women. CLIP produces extreme values of social perception across different ages and facial expressions with the (intersectional) Black women face data. 

We compare our findings with those from the same vision-language association using two public datasets of wild-collected face images. In those datasets, many perceptions of demographic groups are no longer significantly different. This difference from our experimental results implies that uncontrolled variation in wild-collected faces inhibits their statistical ability to detect protected attribute biases, which we can identify more precisely.

## Citation
```bibtex
@article{Hausladen2024,
  title={Measuring social bias in face perception of a vision-language model},
  author={Carina I. Hausladen and Manuel Knott and Colin F. Camerer and Pietro Perona},
  year={2024},
  eprint={arXiv:},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```
## Code
Code will be made available soon.

