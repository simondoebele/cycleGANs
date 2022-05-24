import torch
segm_model = torch.hub.load('pytorch/vision:v0.10.0', 
                            'deeplabv3_resnet50', pretrained=True)
segm_model.eval()


"""All pre-trained models expect input images normalized in the same way, 
i.e. mini-batches of 3-channel RGB images of shape (N, 3, H, W), 
where N is the number of images, H and W are expected to be at least 224 pixels. 
The images have to be loaded in to a range of [0, 1] and then normalized using 
mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. 
output['out'] is of shape (N, 21, H, W)"""

