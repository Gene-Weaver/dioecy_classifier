import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True

class FeatureExtractor:
    """ Class for extracting activations and registering gradients from target intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = {}
        self.activations = {}

        for name, module in self.model.named_modules():
            if name in self.target_layers:
                module.register_forward_hook(self.save_activation(name))
                module.register_backward_hook(self.save_gradient(name))

    def save_activation(self, name):
        def hook(module, input, output):
            self.activations[name] = output
        return hook

    def save_gradient(self, name):
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0]
        return hook

    def get_activations_and_gradients(self, x):
        self.model.zero_grad()
        output = self.model(x)
        return self.activations, output

class GradCAM:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.feature_extractor = FeatureExtractor(model, target_layer_names)

    def __call__(self, input, index=None):
        if self.cuda:
            input = input.cuda()

        activations, output = self.feature_extractor.get_activations_and_gradients(input)

        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = torch.zeros_like(output, device=input.device)
        one_hot[0][index] = 1
        one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.feature_extractor.gradients[self.feature_extractor.target_layers[0]].cpu().data.numpy()
        target = activations[self.feature_extractor.target_layers[0]].cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input.shape[2], input.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


def visualize_activations(model_path, val_dir, save_dir, target_layer_names=['layer4'], use_cuda=True):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load model
    # model = models.resnet50(pretrained=False)
    model = models.resnext101_32x8d(pretrained=False)


    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)  # assuming 7 classes
    # Load the model state dictionary from the checkpoint
    checkpoint = torch.load(model_path)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    if use_cuda:
        model = model.cuda()
    model.eval()

    grad_cam = GradCAM(model=model, target_layer_names=target_layer_names, use_cuda=use_cuda)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Ensure same size as training script
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class_names = sorted(os.listdir(val_dir))

    for class_name in class_names:
        class_dir = os.path.join(val_dir, class_name)
        image_files = os.listdir(class_dir)

        for image_name in tqdm(image_files, desc=f'Processing {class_name}'):
            image_path = os.path.join(class_dir, image_name)
            img = Image.open(image_path).convert('RGB')
            input_img = transform(img).unsqueeze(0)

            # Generate Grad-CAM
            try:
                mask = grad_cam(input_img)
            except RuntimeError as e:
                print(f"Failed to generate Grad-CAM for {image_name}: {e}")
                continue

            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[..., ::-1]  # convert from BGR to RGB


            img = np.array(img) / 255
            heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # Resize heatmap to match img size
            cam_img = heatmap_resized + img
            cam_img = cam_img / np.max(cam_img)

            # Save overlayed image
            save_path = os.path.join(save_dir, f'{class_name}_{image_name}')
            plt.imsave(save_path, cam_img)

if __name__ == "__main__":
    # visualize_activations(
    #     model_path='D:/Dropbox/SwinV2_Classifier/ResNet_v_2/ResNet_v_2.pth',
    #     val_dir='D:/Dropbox/SwinV2_Classifier/data/training_v_1/test',
    #     save_dir='D:/Dropbox/SwinV2_Classifier/ResNet_v_2/Activations_eComplete',
    #     use_cuda=True
    # )
    visualize_activations(
        model_path='D:/Dropbox/SwinV2_Classifier/ResNeXt_v_2_2/ResNeXt_v_2_2.pth',
        val_dir='D:/Dropbox/SwinV2_Classifier/data/training_v_2/train',
        save_dir='D:/Dropbox/SwinV2_Classifier/ResNeXt_v_2_2/Activations__train_eComplete',
        use_cuda=True,

    )
