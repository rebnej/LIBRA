# Model-Agnostic Gender Debiased Image Captioning (CVPR 2023) 
[paper]: https://arxiv.org/abs/2304.03693 
This repository contains source code for the paper titled: "Model-Agnostic Gender Debiased Image Captioning" (Accepted to CVPR 2023). [[paper]] 


## Installation

To set up your environment to run the project, follow these steps:

- **Requirements**:
  - PyTorch 1.11.0 
  - Torchvision 0.12.0
  - CUDA 11.3

- **Install Dependencies**:
  Run the following command to install the necessary packages:

  ```bash
  pip install -r requirements.txt
  python3 -m spacy download en_core_web_sm
  ```

## Usage

### Data preparation

* Download COCO 2014 (train and validation sets) from https://cocodataset.org/. 
* Place .pkl file of the output captions of a image captioning model in `Data`. Please refer to `oscar_preds.pkl` in `Data` for the format.  

### Model preparation

[here]: https://drive.google.com/drive/folders/1OluSU4amjX-RKTXF70RzIntqoCWUxJU7?usp=sharing
Download the trained model from [here] and place it in `Models`.

### Generate Debiased Captions

To generate debiased captions, run the following command:

```python
python gpt2_generate_debiased_cap.py \
--pred_cap_path Data/oscar_preds.pkl \
--model_path Models/libra_final.pth \
--image_dir path/to/coco/val2014/directory \
--rand_test_ipt_mask True
--rand_mask_rate 0.2
```

## Citation

If you use this project in your research or wish to refer to the baseline results published in the paper, please use the following .bib entry.

```bibtex
@inproceedings{hirota2023model,
  title={Model-Agnostic Gender Debiased Image Captioning},
  author={Hirota, Yusuke and Nakashima, Yuta and Garcia, Noa},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15191--15200},
  year={2023}
}
```

## TODOs

- [x] Debiased Caption Generation
- [ ] Training (Upcoming feature)

---

Feel free to contribute to this project or suggest improvements. Your feedback and contributions are greatly appreciated!
