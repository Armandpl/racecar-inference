import torchmetrics
from torchvision.utils import save_image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import PIL

config = dict(
    architecture="resnet34",
    pretrained=True,
    learning_rate=1e-3,
    loss="binary_cross_entropy",
)


class RoadRegressionTurn(nn.Module):

    def __init__(self):
        super().__init__()
        self.config = config

        # setting up metrics
        metrics = torchmetrics.MetricCollection([
            torchmetrics.MeanSquaredError(),
            torchmetrics.MeanAbsoluteError(),
            torchmetrics.Accuracy(),
            torchmetrics.F1Score(),
            torchmetrics.Precision(),
            torchmetrics.Recall(),
        ])
        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')

        self.activation = {}
        
        self.model = self.build_model()
        
        self.classification_head = nn.Linear(86528, 1)

        self.regression_head = self.model.classifier
        self.model.classifier = nn.Identity()


    def forward(self, x):

        features = self.model(x)
        
        brake = self.classification_head(features)
        brake = torch.sigmoid(brake)
        center = self.regression_head(features)


        # return center, brake
        return torch.cat((center, brake), dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.config["learning_rate"])
        return optimizer

    # TODO We shouldn't use a separate function for this.
    def build_model(self):
        model = torchvision.models.squeezenet1_0()
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(86528, 2)
        )
        # load pth file
        # model.load_state_dict(torch.load("checkpoints/road_regression/model-v163.pth"))
        # for param in model.parameters():
        #     param.requires_grad = False
        return model