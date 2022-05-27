import torch
from scipy.ndimage import binary_fill_holes
from PIL import Image
import torchvision.transforms as transforms
from zzsn2021.systems.segmentor import Segmentor
from zzsn2021.configs import Config
from zzsn2021.configs.experiment import ExperimentSettings


def save_output(tensor, img_path):
    to_pil = transforms.ToPILImage()
    changed_to_pil = to_pil(tensor)
    save_path = '/'.join(img_path.split('/')[:-1]) + '/changed_' + img_path.split('/')[-1]
    changed_to_pil.save(save_path)
    changed_to_pil = Image.open(save_path)
    changed_to_pil.show()


def change_background(img, pred, background, to_original_size):
    to_tensor = transforms.ToTensor()
    masked = (to_tensor(img) * pred)

    background = to_tensor(background)
    background = to_original_size(background)
    return torch.where(pred == 0, background, masked)


def predict(model, img, to_original_size):
    to_model_input_size = transforms.Compose([
        transforms.ToTensor(), transforms.Resize(120)
    ])
    img_resized = to_model_input_size(img).unsqueeze(0)
    pred = model(img_resized)
    pred = to_original_size(pred).detach().round().int().squeeze(0).squeeze(0)
    return torch.tensor(binary_fill_holes(pred).astype(int))


def load_model(ckpt_path):
    cfg = Config()
    cfg.experiment = ExperimentSettings(n_classes=1)
    unet = Segmentor.load_from_checkpoint(ckpt_path, cfg=cfg)
    unet.eval()
    return unet


def change_background_pipeline():
    ckpt_path = "../trained_cpkts/unet_11_epochs.cpkt"
    # img_path = '../test_imgs/001bcd05-500.webp'
    img_path = '../test_imgs/pawel.jpg'
    # img_path = '../test_imgs/fetchimage.webp'
    # img_path = '../test_imgs/Clooney_CAASpeakers_Photo.jpg'
    background_path = '../test_imgs/tlo.webp'
    img = Image.open(img_path)
    background = Image.open(background_path)
    img.show()
    background.show()

    to_original_size = transforms.Resize((img.height, img.width))
    unet = load_model(ckpt_path)
    pred = predict(unet, img, to_original_size)
    changed = change_background(img, pred, background, to_original_size)
    save_output(changed, img_path)


if __name__ == '__main__':
    change_background_pipeline()
