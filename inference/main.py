import torch
import io
import os
import time
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, normalize
from PIL import Image

import ray
from ray.data import range_tensor


# Load the trained model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def model_init():
    model = NeuralNetwork()
    model_path = "models/03-05-2023-19-00-41/dict_checkpoint.pkl"
    model = torch.load(model_path)
    return model

model = model_init()

# def predict_from_file(image_path):
#     image = None
#     image = Image.open(image_path)
    
#     # Preprocess the image
#     image = image.convert('L')  # convert to grayscale
#     image = image.resize((28, 28))  # resize
#     image = to_tensor(image)  # convert to tensor
#     print("type: ", type(image))
#     print("shape: ", image.shape)
#     # print(image.shape)

#     with torch.no_grad():
#         output = model(image.unsqueeze(0))
#         prediction = output.argmax(dim=1).item()
    
#     return json.dumps({'prediction': prediction})
# res = predict_from_file("./dataset/7.png")
# print(res)


def predict(image_tensor):
    image_tensor = torch.from_numpy(image_tensor)
    print("type: ", type(image_tensor))
    print("shape: ", image_tensor.shape)
    # image_tensor = torch.from_numpy(image_tensor)
    # print("after type: ", type(image_tensor))
    # print("shape: ", image_tensor.shape)
    with torch.no_grad():
        output = model(image_tensor.squeeze(0))
        prediction = output.argmax(dim=1).item()
    
    res = {'prediction': prediction}
    print(res)
    return [json.dumps(res)]

ctx = ray.data.DatasetContext.get_current()
ctx.execution_options.verbose_progress = True
ctx.execution_options.resource_limits.cpu = 3

for _ in (
    ray.data.range_tensor(1000, shape=(28, 28, 1), parallelism=1000)
    .map_batches(predict, num_cpus=4)
    .iter_batches()
):
    # iterative object
    # do next step
    # or store the result
    pass