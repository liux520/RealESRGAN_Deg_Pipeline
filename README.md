## Real-ESRGAN Degradation Pipeline. 

You can generate your own degraded datasets using this pipeline. There are three modes that can be used here.

### Mode 1: Generate degraded images and save them.

```python
from utils.utils import _get_paths_from_images
from utils.utils import uint2tensor

# print(os.path.abspath(os.path.join(__file__, os.path.pardir)))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
deg_pipeline = Degradation(scale=2, gt_size=480, use_sharp_gt=True, device=device)

""" Demo-1: Generate degraded images and save them. """

# input path: directory or single-image path
input_path = r'../../Datasets/SR/DIV2K/DIV2K_HR_train/0004.png'
save_path = r'figs'
os.makedirs(save_path, exist_ok=True)

if os.path.isdir(input_path):
    gt_paths = _get_paths_from_images(input_path)
else:
    gt_paths = [input_path]

for gt_path in gt_paths:
    base, ext = os.path.splitext(os.path.basename(gt_path))

    gt, gt_usm, lq, kernel1, kernel2, sinc_kernel = deg_pipeline(gt_path, uint8=True, test=True)

    cv2.imwrite(os.path.join(save_path, f'{base}_deg{ext}'), lq[:, :, ::-1])
    cv2.imwrite(os.path.join(save_path, f'{base}_gt{ext}'), gt[:, :, ::-1])
    cv2.imwrite(os.path.join(save_path, f'{base}_gtusm{ext}'), gt_usm[:, :, ::-1])

```

### Mode 2: Accessing the Dataset class via interface *forward_interface*.

```python
from utils.utils import _get_paths_from_images
from utils.utils import uint2tensor

# print(os.path.abspath(os.path.join(__file__, os.path.pardir)))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
deg_pipeline = Degradation(scale=2, gt_size=480, use_sharp_gt=True, device=device)

""" Demo-2: Accessing the Dataset class via interface forward_interface. """

gt, gt_usm, lq, kernel1, kernel2, sinc_kernel = deg_pipeline.forward_interface(
    uint2tensor(cv2.imread(r'../../Datasets/SR/DIV2K/DIV2K_HR_train/0004.png')[:, :, ::-1], device)
)

```

### Mode 3: Accessing the Dataset class via interface *forward_interface_contrast*.

```python
from utils.utils import _get_paths_from_images
from utils.utils import uint2tensor

# print(os.path.abspath(os.path.join(__file__, os.path.pardir)))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
deg_pipeline = Degradation(scale=2, gt_size=480, use_sharp_gt=True, device=device)

""" Demo-3: Accessing the Dataset class via interface forward_interface_contrast,
            achieving the same degradation of two image patches. (For some contrast learning based methods). """

gt1, gt_usm1, lq1, gt2, gt_usm2, lq2 = deg_pipeline.forward_interface_contrast(
    uint2tensor(cv2.imread(r'../../Datasets/SR/DIV2K/DIV2K_HR_train/0004.png')[:, :, ::-1], device),
    uint2tensor(cv2.imread(r'../../Datasets/SR/DIV2K/DIV2K_HR_train/0004.png')[:, :, ::-1], device),
)
print((lq1 == lq2).all())
# tensor(True, device='cuda:0')

```



Note: This Project is derived from https://github.com/xinntao/Real-ESRGAN 

Should you have any question, please create an issue on this repository or contact at liuxmail1220@gmail.com.
