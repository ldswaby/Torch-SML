from typing import Dict

import torch
import torch.nn as nn


class ClassificationModel(nn.Module):
    """_summary_
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        layers = list(model.children())
        self.feature_extractor = nn.Sequential(*layers[:-1])
        self.classifier = layers[-1]

    def forward(self, x) -> Dict[str, torch.Tensor]:
        """Fetches all values

        Args:
            x (_type_): _description_

        Returns:
            Dict[str, torch.Tensor]: _description_
        """
        out = {}

        # Generate embeddings
        out['embeddings'] = self.feature_extractor(x).flatten(1)

        # Get logits
        out['logits'] = self.classifier(out['embeddings'])

        # Compute probabilities and predicted class labels
        out['probs'] = torch.softmax(out['logits'], dim=1)
        out['preds'] = torch.argmax(out['probs'], dim=1)

        return out


# import torch
# from torchvision import models, transforms
# from PIL import Image
# from AML.utils import set_torch_device
# from AML.models import MODEL_REGISTRY

# # Check if MPS is available and set device
# device = set_torch_device()


# # Load and preprocess the image
# image_path = "/Users/lukeswaby/Desktop/CODING/AML/AML/sandbox/istockphoto-1443562748-612x612.jpg"
# image = Image.open(image_path)

# # Define the transformations for ResNet
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
# image_tensor = image_tensor.to(device)

# # Load a pre-trained ResNet model and set it to evaluation mode
# # model = models.resnet50(pretrained=True)


# model = MODEL_REGISTRY.get('resnet50')(pretrained=True)
# model = ClassificationModel(model)
# model = model.to(device)
# model.eval()

# # Perform inference
# with torch.no_grad():
#     output = model(image_tensor)


# # Load the labels for ImageNet
# # from torchvision.models import resnet50, ResNet50_Weights
# # weights = ResNet50_Weights.DEFAULT
# # labels = weights.meta["categories"]
# # predicted_label = labels[predicted_idx.item()]

# print(output)

# from torchvision.models import resnet50, ResNet50_Weights
# weights = ResNet50_Weights.DEFAULT
# labels = weights.meta["categories"]
# predicted_label = labels[output['preds'].item()]


# print(f"Predicted label: {predicted_label}")
