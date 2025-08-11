# nails

This repository contains the code for the paper: **[Normative Alignment of Recommender Systems via Internal Label Shift](https://doi.org/10.1145/3705328.3759309)**, accepted as an *extended abstract* at **[RecSys '25](https://recsys.acm.org/) Late Breaking Results**.  

---

## Model & Implementation Details

We use the **NRMSDocVec** model from the [ebnerd-benchmark](https://github.com/ebanalyse/ebnerd-benchmark) repository.  

⚠ **Important note:** Our setup expects that the model outputs logits or prediction scores that sum to 1. We use softmax in the scripts. 

Therefore, in our experiments, we commented out the following line in [`nrms_docvec.py` (line 183)](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/src/ebrec/models/newsrec/nrms_docvec.py):
```python
pred_one = tf.keras.layers.Activation(activation="sigmoid")(pred_one)
```
---

## Download Prediction Files

We share the prediction scores obtained from training, which we use to generate the results: [**Download here**](https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/unrelated/nails_data.zip)

---

## Data Format

The shared prediction file is stored in tabular form with the following structure:

- **Shape:** `(13,536,710, 3)`  
- **Columns:**
  - `impression_id` *(u32)* — Unique identifier for the impression.
  - `article_ids_inview` *(list[i32])* — List of article IDs shown in the impression.
  - `scores` *(list[f32])* — Corresponding model prediction scores for the articles in `article_ids_inview`.

**Example:**

| impression_id | article_ids_inview               | scores                                 |
|---------------|----------------------------------|----------------------------------------|
| 10017530      | [9794425, 9794706, ..., 9794673] | [-0.013472, -0.458544, ..., 0.873...]  |
| 28473735      | [9794845, 9794924, ..., 9794932] | [0.886794, -0.158117, ..., 0.5009...]  |
| 32426821      | [9797023, 9798775, ..., 9798644] | [0.802967, 0.521927, ..., 0.35544...]  |
| 28680972      | [9791182, 9789674, ..., 9756075] | [-0.181093, 0.288003, ..., -2.459...]  |
| 12308406      | [9797733, 9797537, ..., 9798323] | [0.044418, -0.334013, ..., 0.4230...]  |
| ...           | ...                              | ...                                    |


---

## Setup
```bash
conda create -n nails python=3.11
conda activate nails
pip install -r requirements.txt
```

---

## Run Experiments

**Editorial distribution:**
```bash
python exp_nails.py --distribution_type Editorial
python exp_steck.py --distribution_type Editorial
python exp_nails_steck_combine.py --distribution_type Editorial
```

**Uniform distribution:**
```bash
python exp_nails.py --distribution_type Uniform
python exp_steck.py --distribution_type Uniform
python exp_nails_steck_combine.py --distribution_type Uniform
```
---

## Quick Dummy Run
```bash
python exp_nails.py --n_samples 150 --n_samples_test 151
python exp_steck.py --n_samples 150 --n_samples_test 151
python exp_nails_steck_combine.py --n_samples 150
```