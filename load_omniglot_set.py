import learn2learn as l2l
from PIL.Image import LANCZOS
from torchvision import transforms
import random

def load_omniglot(ways, shots):
    omniglot = l2l.vision.datasets.FullOmniglot(root='~/data',
                                                transform=transforms.Compose([
                                                    transforms.Resize(28, interpolation=LANCZOS),
                                                    transforms.ToTensor(),
                                                    lambda x: 1.0 - x,
                                                ]),
                                                download=True)
    dataset = l2l.data.MetaDataset(omniglot)
    classes = list(range(1623))
    random.shuffle(classes)

    train_transforms = [
        l2l.data.transforms.FusedNWaysKShots(dataset,
                                             n=ways,
                                             k=2*shots,
                                             filter_labels=classes[:1200]),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
        l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
    ]
    train_tasks = l2l.data.TaskDataset(dataset,
                                       task_transforms=train_transforms,
                                       num_tasks=20000)

    # valid_transforms = [
    #     l2l.data.transforms.FusedNWaysKShots(dataset,
    #                                          n=ways,
    #                                          k=2*shots,
    #                                          filter_labels=classes[1100:1200]),
    #     l2l.data.transforms.LoadData(dataset),
    #     l2l.data.transforms.RemapLabels(dataset),
    #     l2l.data.transforms.ConsecutiveLabels(dataset),
    #     l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
    # ]
    # valid_tasks = l2l.data.TaskDataset(dataset,
    #                                    task_transforms=valid_transforms,
    #                                    num_tasks=1024)

    test_transforms = [
        l2l.data.transforms.FusedNWaysKShots(dataset,
                                             n=ways,
                                             k=2*shots,
                                             filter_labels=classes[1200:]),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
        l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
    ]
    test_tasks = l2l.data.TaskDataset(dataset,
                                      task_transforms=test_transforms,
                                      num_tasks=1024)

    return train_tasks, test_tasks