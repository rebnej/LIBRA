# Model-Agnostic Gender Debiased Image Captioning (CVPR 2023) 
[paper]: https://openaccess.thecvf.com/content/CVPR2023/papers/Hirota_Model-Agnostic_Gender_Debiased_Image_Captioning_CVPR_2023_paper.pdf
This repository contains source code for the paper titled: "Model-Agnostic Gender Debiased Image Captioning" (Accepted to CVPR 2023). [paper] 


## Installation

To set up your environment to run the project, follow these steps:

- **Requirements**:
  - PyTorch (specify version)
  - CUDA (specify version)

- **Install Dependencies**:
  Run the following command to install the necessary packages:

  ```bash
  pip install -r requirements.txt
  ```

## Usage

### Generate Debiased Captions

To generate debiased captions, run the following command:

```python
python generate_debiased_captions.py [add your parameters here]
```

Replace `[add your parameters here]` with the appropriate parameters for your script.

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
