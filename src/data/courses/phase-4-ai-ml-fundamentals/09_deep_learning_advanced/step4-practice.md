# Advanced Deep Learning Practice

## Table of Contents

1. [Implementing ResNet from Scratch](#implementing-resnet-from-scratch)
2. [Transfer Learning Implementation](#transfer-learning-implementation)
3. [Advanced Regularization Techniques](#advanced-regularization-techniques)
4. [Model Evaluation and Debugging](#model-evaluation-and-debugging)
5. [Hyperparameter Tuning Strategies](#hyperparameter-tuning-strategies)
6. [Building Ensemble Models](#building-ensemble-models)
7. [Performance Optimization](#performance-optimization)
8. [Real-world Project Implementations](#real-world-project-implementations)

---

## Implementing ResNet from Scratch

### ResNet Architecture Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Basic ResNet block with two 3x3 convolutions"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    """Bottleneck ResNet block with 1x1, 3x3, 1x1 convolutions"""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1,
                              bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                              bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet blocks
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# Factory functions for different ResNet variants
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])
```

### ResNet Training Implementation

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

class ResNetTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                                   weight_decay=1e-4)
        self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        accuracy = 100. * correct / total
        avg_loss = running_loss / len(train_loader)
        return avg_loss, accuracy

    def validate(self, val_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        test_loss /= len(val_loader)
        accuracy = 100. * correct / total
        return test_loss, accuracy

    def train(self, train_loader, val_loader, epochs):
        best_acc = 0
        patience = 10
        counter = 0

        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print('-' * 50)

            # Early stopping
            if val_acc > best_acc:
                best_acc = val_acc
                counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_resnet.pth')
            else:
                counter += 1
                if counter >= patience:
                    print(f'Early stopping after {epoch+1} epochs')
                    break

            self.scheduler.step()

# Training script
def train_resnet():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                    download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                   download=True, transform=transform_test)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    # Model and trainer
    model = ResNet18()
    trainer = ResNetTrainer(model, device)

    # Train
    trainer.train(train_loader, test_loader, epochs=200)

# train_resnet()
```

---

## Transfer Learning Implementation

### Feature Extraction Approach

```python
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader

class TransferLearningModel:
    def __init__(self, num_classes, model_name='resnet50', feature_extract=True):
        self.model_name = model_name
        self.num_classes = num_classes
        self.feature_extract = feature_extract

        self.model = self._initialize_model()
        self.set_parameter_requires_grad(self.feature_extract)
        self._update_fc_layer()

    def _initialize_model(self):
        # Initialize these variables which will be set in this function
        model = None
        input_size = 0

        if self.model_name == "resnet":
            model = models.resnet50(pretrained=True)
            input_size = 224
        elif self.model_name == "alexnet":
            model = models.alexnet(pretrained=True)
            input_size = 224
        elif self.model_name == "vgg":
            model = models.vgg11_bn(pretrained=True)
            input_size = 224
        elif self.model_name == "densenet":
            model = models.densenet121(pretrained=True)
            input_size = 224
        elif self.model_name == "inception":
            model = models.inception_v3(pretrained=True)
            input_size = 299
        else:
            print("Invalid model name, exiting...")
            exit()

        return model, input_size

    def set_parameter_requires_grad(self, feature_extracting):
        if self.feature_extract:
            for param in self.model.parameters():
                param.requires_grad = False

    def _update_fc_layer(self):
        if self.model_name == "resnet":
            self.model.fc = nn.Linear(2048, self.num_classes)
        elif self.model_name == "alexnet":
            self.model.classifier[6] = nn.Linear(4096, self.num_classes)
        elif self.model_name == "vgg":
            self.model.classifier[6] = nn.Linear(4096, self.num_classes)
        elif self.model_name == "densenet":
            self.model.classifier = nn.Linear(1024, self.num_classes)
        elif self.model_name == "inception":
            self.model.fc = nn.Linear(2048, self.num_classes)

    def get_parameter_groups(self):
        params_to_update = self.model.parameters()

        if self.feature_extract:
            params_to_update = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
        else:
            params_to_update = self.model.parameters()

        return params_to_update

def train_transfer_learning(data_dir, num_classes, epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data augmentation and normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                             data_transforms[x])
                     for x in ['train', 'val']}

    # Create data loaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                       batch_size=8,
                                                       shuffle=True,
                                                       num_workers=4)
                       for x in ['train', 'val']}

    # Initialize model
    model = TransferLearningModel(num_classes, model_name='resnet50',
                                 feature_extract=True)
    model = model.model.to(device)

    # Optimizer with different learning rates
    params_to_update = model.get_parameter_groups()
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Train model
    best_acc = 0.0

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 30)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders_dict[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model
```

### Progressive Unfreezing Implementation

```python
class ProgressiveUnfreezingTrainer:
    def __init__(self, model, base_model, unfreeze_layers, num_classes):
        self.model = model
        self.base_model = base_model
        self.unfreeze_layers = unfreeze_layers
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initially freeze all layers except the final classifier
        self.freeze_all_layers()
        self.unfreeze_final_classifier()

    def freeze_all_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_final_classifier(self):
        self.model.fc = nn.Linear(2048, self.num_classes)
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def unfreeze_layer_groups(self, layer_groups):
        """Unfreeze specific layer groups"""
        if layer_groups == 'layer4':
            for param in self.model.layer4.parameters():
                param.requires_grad = True
        elif layer_groups == 'layer3':
            for param in self.model.layer3.parameters():
                param.requires_grad = True
        elif layer_groups == 'layer2':
            for param in self.model.layer2.parameters():
                param.requires_grad = True
        elif layer_groups == 'layer1':
            for param in self.model.layer1.parameters():
                param.requires_grad = True
        elif layer_groups == 'all':
            for param in self.model.parameters():
                param.requires_grad = True

    def get_optimizer_with_different_lrs(self, base_lr=0.001):
        """Get optimizer with different learning rates for different layers"""
        # Different learning rates for different layer groups
        param_groups = [
            {'params': self.model.fc.parameters(), 'lr': base_lr * 10},
        ]

        # Add progressively lower learning rates for earlier layers
        if self.model.layer4[0].conv1.weight.requires_grad:
            param_groups.append({'params': self.model.layer4.parameters(),
                               'lr': base_lr * 1})
        if self.model.layer3[0].conv1.weight.requires_grad:
            param_groups.append({'params': self.model.layer3.parameters(),
                               'lr': base_lr * 0.1})
        if self.model.layer2[0].conv1.weight.requires_grad:
            param_groups.append({'params': self.model.layer2.parameters(),
                               'lr': base_lr * 0.01})
        if self.model.layer1[0].conv1.weight.requires_grad:
            param_groups.append({'params': self.model.layer1.parameters(),
                               'lr': base_lr * 0.001})
        if self.model.conv1.weight.requires_grad:
            param_groups.append({'params': self.model.conv1.parameters(),
                               'lr': base_lr * 0.0001})

        return optim.Adam(param_groups, lr=base_lr)

    def progressive_training(self, train_loader, val_loader, total_epochs=50):
        model = self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()

        # Training phases
        phases = [
            {'unfreeze': 'final_classifier', 'epochs': 5, 'base_lr': 0.001},
            {'unfreeze': 'layer4', 'epochs': 10, 'base_lr': 0.0001},
            {'unfreeze': 'layer3', 'epochs': 10, 'base_lr': 0.00001},
            {'unfreeze': 'layer2', 'epochs': 10, 'base_lr': 0.000001},
            {'unfreeze': 'layer1', 'epochs': 10, 'base_lr': 0.0000001},
            {'unfreeze': 'all', 'epochs': 5, 'base_lr': 0.00000001}
        ]

        best_acc = 0.0

        for phase_config in phases:
            print(f"Training with {phase_config['unfreeze']} unfrozen")

            # Unfreeze appropriate layers
            if phase_config['unfreeze'] != 'final_classifier':
                self.unfreeze_layer_groups(phase_config['unfreeze'])

            # Get optimizer with appropriate learning rates
            optimizer = self.get_optimizer_with_different_lrs(phase_config['base_lr'])

            # Train this phase
            for epoch in range(phase_config['epochs']):
                self.train_epoch(model, train_loader, optimizer, criterion)
                val_acc = self.validate(model, val_loader, criterion)

                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), 'best_progressive_model.pth')

                print(f"Phase {phase_config['unfreeze']}, Epoch {epoch+1}, "
                      f"Val Acc: {val_acc:.4f}")

        return model

    def train_epoch(self, model, train_loader, optimizer, criterion):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        return running_loss / len(train_loader), accuracy

    def validate(self, model, val_loader, criterion):
        model.eval()
        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        return accuracy
```

---

## Advanced Regularization Techniques

### Custom Dropout Layers

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DropBlock(nn.Module):
    """DropBlock: A structured form of dropout for conv layers"""
    def __init__(self, block_size, p=0.1, training=True):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.p = p
        self.training = training

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        if x.dim() != 4:
            raise ValueError(f"DropBlock requires 4D input, got {x.dim()}D")

        # Get batch size, height, width
        N, C, H, W = x.shape

        # Calculate the actual dropout probability
        gamma = (self.p * H * W) / ((self.block_size ** 2) * (H - self.block_size + 1) ** 2)

        # Create mask
        mask = torch.ones((N, C, H, W), device=x.device)
        mask = F.dropout2d(mask, p=self.p, training=True)

        # Apply block mask
        for n in range(N):
            for c in range(C):
                if torch.rand(1) < gamma:
                    h, w = torch.randint(0, H - self.block_size, (2,))
                    mask[n, c, h:h+self.block_size, w:w+self.block_size] = 0

        # Normalize the mask
        mask_count = (mask.sum() / (N * C))
        mask = mask / mask_count

        return x * mask

class ScheduledDropout(nn.Module):
    """Dropout with schedule that increases dropout rate during training"""
    def __init__(self, initial_p=0.1, final_p=0.5, total_steps=10000, start_step=1000):
        super(ScheduledDropout, self).__init__()
        self.initial_p = initial_p
        self.final_p = final_p
        self.total_steps = total_steps
        self.start_step = start_step
        self.current_step = 0

    def forward(self, x):
        if not self.training:
            return x

        self.current_step += 1

        if self.current_step < self.start_step:
            p = self.initial_p
        else:
            progress = min(1.0, (self.current_step - self.start_step) /
                          (self.total_steps - self.start_step))
            p = self.initial_p + (self.final_p - self.initial_p) * progress

        return F.dropout(x, p=p, training=True)

class EfficientDropout(nn.Module):
    """Memory-efficient dropout for large models"""
    def __init__(self, p=0.5):
        super(EfficientDropout, self).__init__()
        self.p = p
        self.mask = None
        self.input_shape = None

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        # Create or update mask
        if self.mask is None or self.input_shape != x.shape:
            self.input_shape = x.shape
            prob = 1 - self.p
            self.mask = (torch.rand(x.shape, device=x.device) < prob) / prob
        else:
            # Update existing mask
            prob = 1 - self.p
            self.mask = (torch.rand(x.shape, device=x.device) < prob) / prob

        return x * self.mask

# Usage example
class RegularizedCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super(RegularizedCNN, self).__init__()

        # Different dropout strategies
        self.drop_block = DropBlock(block_size=3, p=0.1)
        self.scheduled_dropout = ScheduledDropout(initial_p=0.1, final_p=0.3)
        self.efficient_dropout = EfficientDropout(p=0.2)

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.DropBlock(block_size=3, p=0.1),  # Structured dropout

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            self.drop_block,  # Another structured dropout
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            self.scheduled_dropout,  # Scheduled dropout
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

### Advanced Batch Normalization Techniques

```python
class GhostBatchNorm(nn.Module):
    """Ghost Batch Normalization for small batch training"""
    def __init__(self, num_features, num_splits=32, eps=1e-5, momentum=0.1):
        super(GhostBatchNorm, self).__init__()
        self.num_features = num_features
        self.num_splits = num_splits
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if self.training:
            N, C, H, W = x.shape

            # Split batch into multiple smaller batches
            batch_size = N // self.num_splits
            if batch_size == 0:
                return self._bn_single(x)

            outputs = []
            for i in range(self.num_splits):
                start_idx = i * batch_size
                if i == self.num_splits - 1:  # Last split takes remaining samples
                    end_idx = N
                else:
                    end_idx = (i + 1) * batch_size

                split_x = x[start_idx:end_idx]
                output = self._bn_single(split_x)
                outputs.append(output)

            return torch.cat(outputs, dim=0)
        else:
            return self._bn_single(x)

    def _bn_single(self, x):
        # Apply batch normalization
        return F.batch_norm(x, self.running_mean, self.running_var,
                          self.weight, self.bias, self.training, self.momentum, self.eps)

class AdaptiveBatchNorm(nn.Module):
    """Adaptive Batch Normalization that adjusts normalization statistics"""
    def __init__(self, num_features, adapt_momentum=0.1):
        super(AdaptiveBatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.adapt_momentum = adapt_momentum

        # Adapter network
        self.adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_features * 2)  # γ and β
        )

    def forward(self, x):
        # Standard batch norm
        bn_output = self.bn(x)

        if self.training:
            # Compute adaptation parameters
            adapt_params = self.adapter(x)
            gamma_adapt = adapt_params[:, :x.size(1)]
            beta_adapt = adapt_params[:, x.size(1):x.size(1)*2]

            # Apply adaptation
            output = gamma_adapt.view(x.size(0), x.size(1), 1, 1) * bn_output + \
                    beta_adapt.view(x.size(0), x.size(1), 1, 1)
            return output
        else:
            return bn_output

class ConditionalBatchNorm(nn.Module):
    """Conditional Batch Normalization using class embeddings"""
    def __init__(self, num_features, num_classes):
        super(ConditionalBatchNorm, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)

        # Class-conditional parameters
        self.embedding = nn.Embedding(num_classes, num_features * 2)
        self.embedding.weight.data.normal_(0, 0.02)

    def forward(self, x, class_labels):
        # Standard batch norm
        bn_output = self.bn(x)

        # Get class-conditional parameters
        gamma_beta = self.embedding(class_labels)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)
        beta = beta.view(x.size(0), x.size(1), 1, 1)

        # Apply conditional scaling and shifting
        output = gamma * bn_output + beta
        return output
```

---

## Model Evaluation and Debugging

### Training Diagnostics

```python
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class TrainingDiagnostics:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.gradient_norms = []

    def track_training_metrics(self, train_loss, train_acc, val_loss, val_acc,
                              learning_rate, gradient_norm):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(learning_rate)
        self.gradient_norms.append(gradient_norm)

    def plot_training_curves(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Training Loss')
        axes[0, 0].plot(self.val_losses, label='Validation Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy curves
        axes[0, 1].plot(self.train_accuracies, label='Training Accuracy')
        axes[0, 1].plot(self.val_accuracies, label='Validation Accuracy')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Learning rate schedule
        axes[1, 0].plot(self.learning_rates)
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)

        # Gradient norms
        axes[1, 1].plot(self.gradient_norms)
        axes[1, 1].set_title('Gradient Norms')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Gradient Norm')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig('training_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_gradient_flow(self, verbose=True):
        """Analyze gradient flow through the network"""
        self.model.train()

        # Create a dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        dummy_output = self.model(dummy_input)
        target = torch.randint(0, 1000, (1,)).to(self.device)

        # Compute loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(dummy_output, target)

        # Backward pass
        loss.backward()

        gradient_info = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                gradient_info.append({
                    'layer': name,
                    'grad_norm': grad_norm,
                    'param_norm': param.data.norm(2).item(),
                    'grad_param_ratio': grad_norm / (param.data.norm(2).item() + 1e-8)
                })

        if verbose:
            print("Gradient Flow Analysis:")
            print("-" * 80)
            print(f"{'Layer':<50} {'Grad Norm':<12} {'Param Norm':<12} {'Ratio':<8}")
            print("-" * 80)

            for info in gradient_info:
                print(f"{info['layer']:<50} {info['grad_norm']:<12.6f} "
                      f"{info['param_norm']:<12.6f} {info['grad_param_ratio']:<8.4f}")

        return gradient_info

    def identify_gradient_problems(self):
        """Identify potential gradient problems"""
        gradient_info = self.analyze_gradient_flow(verbose=False)

        problems = []

        # Check for exploding gradients
        exploding_layers = [info for info in gradient_info if info['grad_norm'] > 100]
        if exploding_layers:
            problems.append({
                'type': 'exploding_gradients',
                'severity': 'high',
                'description': f'Found {len(exploding_layers)} layers with very large gradients',
                'layers': exploding_layers
            })

        # Check for vanishing gradients
        vanishing_layers = [info for info in gradient_info if info['grad_norm'] < 1e-6]
        if len(vanishing_layers) > len(gradient_info) * 0.5:
            problems.append({
                'type': 'vanishing_gradients',
                'severity': 'high',
                'description': f'Found {len(vanishing_layers)} layers with very small gradients',
                'layers': vanishing_layers
            })

        # Check for gradient variance
        if len(gradient_info) > 1:
            grad_norms = [info['grad_norm'] for info in gradient_info]
            grad_variance = np.var(grad_norms)
            if grad_variance > 1000:
                problems.append({
                    'type': 'high_gradient_variance',
                    'severity': 'medium',
                    'description': f'High variance in gradient norms: {grad_variance:.2f}'
                })

        return problems

class ModelAnalyzer:
    def __init__(self, model, test_loader, device='cuda'):
        self.model = model
        self.test_loader = test_loader
        self.device = device

    def evaluate_model(self):
        """Comprehensive model evaluation"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_confidences = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                # Get predictions and confidences
                confidences = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())

        return np.array(all_labels), np.array(all_preds), np.array(all_confidences)

    def plot_confusion_matrix(self, class_names):
        """Plot confusion matrix"""
        labels, preds, _ = self.evaluate_model()

        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_per_class_performance(self, class_names):
        """Analyze per-class performance metrics"""
        labels, preds, _ = self.evaluate_model()

        report = classification_report(labels, preds, target_names=class_names,
                                     output_dict=True)

        # Convert to DataFrame for easy analysis
        class_metrics = []
        for class_name in class_names:
            if class_name in report:
                class_metrics.append({
                    'Class': class_name,
                    'Precision': report[class_name]['precision'],
                    'Recall': report[class_name]['recall'],
                    'F1-Score': report[class_name]['f1-score'],
                    'Support': report[class_name]['support']
                })

        return pd.DataFrame(class_metrics)

    def analyze_confidence_distribution(self, correct_only=True):
        """Analyze prediction confidence distribution"""
        labels, preds, confidences = self.evaluate_model()

        if correct_only:
            # Only consider correct predictions
            correct_mask = labels == preds
            confidences = confidences[correct_mask]
            labels = labels[correct_mask]
            print(f"Analyzing confidence for {np.sum(correct_mask)} correct predictions")
        else:
            print(f"Analyzing confidence for all {len(labels)} predictions")

        # Plot confidence distribution
        max_confidences = np.max(confidences, axis=1)

        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.hist(max_confidences, bins=30, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Maximum Prediction Confidence')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        # Per-class confidence
        plt.subplot(1, 2, 2)
        unique_labels = np.unique(labels)
        for label in unique_labels:
            class_mask = labels == label
            class_confidences = max_confidences[class_mask]
            plt.hist(class_confidences, alpha=0.5, label=f'Class {label}',
                    bins=20, density=True)

        plt.title('Confidence Distribution by Class')
        plt.xlabel('Confidence')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        return max_confidences
```

---

## Hyperparameter Tuning Strategies

### Bayesian Optimization for Hyperparameters

```python
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

class BayesianHyperparameterTuner:
    def __init__(self, model_fn, data_loaders, device='cuda'):
        self.model_fn = model_fn
        self.train_loader, self.val_loader = data_loaders
        self.device = device

    def objective(self, trial):
        # Suggest hyperparameters
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])

        # Model architecture hyperparameters
        hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256, 512])
        num_layers = trial.suggest_int('num_layers', 1, 4)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)

        # Create model
        model = self.model_fn(hidden_size=hidden_size, num_layers=num_layers,
                             dropout_rate=dropout_rate)
        model = model.to(self.device)

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for epoch in range(5):  # Short training for hyperparameter search
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if len(self.train_loader) < 50:  # Limit batches for speed
                    break

                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        # Validation accuracy
        val_acc = self.evaluate_model(model, self.val_loader)

        return val_acc

    def evaluate_model(self, model, data_loader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        accuracy = correct / total
        return accuracy

    def tune(self, n_trials=100, n_jobs=1):
        """Run Bayesian optimization for hyperparameter tuning"""
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs)

        print(f"Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print(f"  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        return study

# Advanced hyperparameter search with different strategies
class HyperparameterSearch:
    def __init__(self, model_fn, data_loaders, device='cuda'):
        self.model_fn = model_fn
        self.train_loader, self.val_loader = data_loaders
        self.device = device

    def random_search(self, param_distributions, n_trials=50):
        """Random search for hyperparameters"""
        results = []

        for trial in range(n_trials):
            # Sample random parameters
            params = {}
            for param_name, param_config in param_distributions.items():
                if param_config['type'] == 'float':
                    params[param_name] = np.random.uniform(
                        param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = np.random.randint(
                        param_config['low'], param_config['high'] + 1
                    )
                elif param_config['type'] == 'choice':
                    params[param_name] = np.random.choice(param_config['choices'])

            # Train and evaluate
            acc = self.train_and_evaluate(params)
            results.append((params, acc))

        # Sort by accuracy
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def grid_search(self, param_grid):
        """Grid search for hyperparameters"""
        results = []

        # Generate all combinations
        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for params in combinations:
            acc = self.train_and_evaluate(params)
            results.append((params, acc))

        # Sort by accuracy
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def train_and_evaluate(self, params):
        """Train model with given hyperparameters and return validation accuracy"""
        try:
            # Create model
            model = self.model_fn(**params)
            model = model.to(self.device)

            # Optimizer
            optimizer = optim.Adam(model.parameters(), lr=params.get('lr', 0.001))
            criterion = nn.CrossEntropyLoss()

            # Short training
            for epoch in range(10):
                self.train_epoch(model, self.train_loader, optimizer, criterion)

            # Evaluate
            val_acc = self.evaluate_model(model, self.val_loader)
            return val_acc

        except Exception as e:
            print(f"Error with params {params}: {e}")
            return 0.0

    def train_epoch(self, model, train_loader, optimizer, criterion):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 100:  # Limit for speed
                break

            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    def evaluate_model(self, model, data_loader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        return correct / total
```

---

## Building Ensemble Models

### Ensemble Implementation

```python
class EnsembleModel(nn.Module):
    """Ensemble of multiple models with voting strategies"""
    def __init__(self, models, voting='soft', dropout_rate=0.0):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.voting = voting
        self.dropout_rate = dropout_rate

        # Monte Carlo dropout during inference
        self.mc_samples = 10 if dropout_rate > 0 else 1

    def forward(self, x):
        predictions = []

        if self.dropout_rate > 0 and self.training:
            # Training with dropout (already enabled)
            for model in self.models:
                pred = model(x)
                predictions.append(pred)
        elif self.dropout_rate > 0 and not self.training:
            # MC-Dropout during inference
            for _ in range(self.mc_samples):
                for model in self.models:
                    pred = model(x)
                    predictions.append(pred)
        else:
            # Standard ensemble
            for model in self.models:
                pred = model(x)
                predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)

        if self.voting == 'soft':
            # Average probabilities
            if predictions.dim() == 3:  # [models, batch, classes]
                ensemble_pred = torch.mean(predictions, dim=0)
            else:
                ensemble_pred = torch.mean(predictions, dim=0)
        elif self.voting == 'hard':
            # Majority vote
            if predictions.dim() == 3:
                _, predicted = torch.max(predictions, dim=2)
            else:
                _, predicted = torch.max(predictions, dim=1)

            # Convert to one-hot for majority vote
            batch_size, num_classes = predictions.shape[1], predictions.shape[2]
            ensemble_pred = torch.zeros(batch_size, num_classes)

            for i in range(predicted.shape[0]):
                for j in range(predicted.shape[1]):
                    ensemble_pred[j, predicted[i, j]] += 1

        return ensemble_pred

def create_ensemble(model_configs, device='cuda'):
    """Create ensemble from multiple model configurations"""
    models = []

    for config in model_configs:
        if config['type'] == 'resnet18':
            model = ResNet18()
        elif config['type'] == 'resnet50':
            model = ResNet50()
        elif config['type'] == 'densenet121':
            from torchvision.models import densenet121
            model = densenet121(pretrained=True)
            # Replace final layer
            model.classifier = nn.Linear(1024, config['num_classes'])

        # Add dropout to final layers
        if 'dropout' in config and config['dropout'] > 0:
            # Add dropout layer before final classification
            if hasattr(model, 'fc'):
                old_fc = model.fc
                model.fc = nn.Sequential(
                    nn.Dropout(config['dropout']),
                    old_fc
                )

        model = model.to(device)
        models.append(model)

    return models

class SnapshotEnsemble:
    """Snapshot ensemble that saves models during training"""
    def __init__(self, model_fn, device='cuda', patience=5):
        self.device = device
        self.patience = patience
        self.snapshots = []
        self.current_model = model_fn().to(device)
        self.best_acc = 0
        self.wait = 0

    def train(self, train_loader, val_loader, epochs, optimizer_fn, criterion):
        self.snapshots = []
        best_model_state = None

        for epoch in range(epochs):
            # Train current epoch
            self.current_model.train()
            running_loss = 0.0

            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer = optimizer_fn(self.current_model.parameters())

                optimizer.zero_grad()
                output = self.current_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Validate
            val_acc = self.validate(self.current_model, val_loader)

            print(f"Epoch {epoch+1}/{epochs}, Val Acc: {val_acc:.4f}")

            # Save snapshot if validation accuracy improves
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.wait = 0
                best_model_state = self.current_model.state_dict().copy()
            else:
                self.wait += 1

            # Save snapshot
            snapshot = self.current_model.state_dict().copy()
            self.snapshots.append({
                'epoch': epoch + 1,
                'state_dict': snapshot,
                'val_acc': val_acc
            })

            # Early stopping
            if self.wait >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Return best model state
        if best_model_state:
            self.current_model.load_state_dict(best_model_state)

        return self.create_ensemble()

    def validate(self, model, val_loader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        return correct / total

    def create_ensemble(self):
        """Create ensemble from all snapshots"""
        if not self.snapshots:
            return None

        # Select top K snapshots
        sorted_snapshots = sorted(self.snapshots, key=lambda x: x['val_acc'], reverse=True)
        top_snapshots = sorted_snapshots[:min(3, len(sorted_snapshots))]

        models = []
        for snapshot in top_snapshots:
            # Recreate model (you would need to store model architecture)
            # For now, we'll assume we have the original model
            model = self.create_model()  # This should be implemented
            model.load_state_dict(snapshot['state_dict'])
            model.eval()
            models.append(model)

        ensemble = EnsembleModel(models, voting='soft')
        return ensemble
```

---

## Performance Optimization

### Memory Optimization Techniques

```python
import torch
import gc
from contextlib import contextmanager

@contextmanager
def torch_no_grad():
    with torch.no_grad():
        yield

class MemoryEfficientTraining:
    """Training utilities for memory-efficient training"""

    @staticmethod
    def gradient_checkpointing(forward_pass_func, *inputs):
        """Implement gradient checkpointing to save memory"""
        def checkpoint(*inputs):
            return torch.utils.checkpoint.checkpoint(forward_pass_func, *inputs)

        return checkpoint

    @staticmethod
    def clear_memory():
        """Clear GPU memory"""
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def get_memory_usage():
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3  # GB
            return {
                'allocated': allocated,
                'cached': cached,
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**3
            }
        return None

    @staticmethod
    def optimize_memory_layout(model, use_channels_last=True):
        """Optimize memory layout for better performance"""
        if use_channels_last:
            for module in model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    if hasattr(module.weight, 'data'):
                        module.weight.data = module.weight.data.contiguous(
                            memory_format=torch.channels_last
                        )
        return model

class MixedPrecisionTrainer:
    """Mixed precision training implementation"""
    def __init__(self, model, optimizer, loss_fn, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler()

    def train_step(self, inputs, targets):
        # Enable auto mixed precision
        with torch.cuda.amp.autocast():
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

        # Scale the loss and backward pass
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        return loss.item()

    def validate_step(self, inputs, targets):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
        return loss.item()

class GradientAccumulation:
    """Gradient accumulation for effective large batch training"""
    def __init__(self, model, optimizer, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.step_count = 0

    def zero_grad(self):
        self.optimizer.zero_grad()
        self.step_count = 0

    def step(self, loss, retain_graph=False):
        # Scale loss by accumulation steps
        loss = loss / self.accumulation_steps
        loss.backward(retain_graph=retain_graph)

        self.step_count += 1

        if self.step_count % self.accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

# Memory-efficient data loading
class MemoryEfficientDataLoader:
    def __init__(self, dataset, batch_size, device='cuda', pin_memory=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.pin_memory = pin_memory and device.type == 'cuda'

        # Create DataLoader with memory optimization
        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )

    def __iter__(self):
        for batch in self.loader:
            # Move data to device efficiently
            if self.pin_memory:
                data = batch[0].pin_memory().to(self.device, non_blocking=True)
                targets = batch[1].pin_memory().to(self.device, non_blocking=True)
            else:
                data, targets = batch
                data = data.to(self.device)
                targets = targets.to(self.device)

            yield data, targets

    def __len__(self):
        return len(self.loader)
```

### Model Compilation and Optimization

```python
import torch.jit
import torch_tensorrt

class ModelOptimizer:
    """Model optimization for production deployment"""

    @staticmethod
    def torch_jit_trace(model, example_input):
        """Use TorchScript to trace and optimize model"""
        model.eval()
        traced_model = torch.jit.trace(model, example_input)
        return traced_model

    @staticmethod
    def torch_jit_script(model):
        """Use TorchScript to script model (supports control flow)"""
        model.eval()
        scripted_model = torch.jit.script(model)
        return scripted_model

    @staticmethod
    def convert_to_tensorrt(model, inputs):
        """Convert PyTorch model to TensorRT for GPU inference"""
        try:
            # First trace the model
            traced_model = torch.jit.trace(model, inputs)

            # Convert to TensorRT (requires TensorRT and torch_tensorrt)
            trt_model = torch_tensorrt.compile(traced_model,
                                             inputs=[inputs],
                                             enabled_precisions={torch.float, torch.half})
            return trt_model
        except Exception as e:
            print(f"TensorRT conversion failed: {e}")
            return None

    @staticmethod
    def optimize_for_inference(model, example_input):
        """Apply various optimizations for inference"""
        model.eval()

        # Set model to evaluation mode
        model.eval()

        # Apply TorchScript optimization
        try:
            optimized_model = torch.jit.trace(model, example_input)
            return optimized_model
        except Exception as e:
            print(f"TorchScript optimization failed: {e}")
            return model

    @staticmethod
    def benchmark_model(model, example_input, num_runs=1000, warmup_runs=100):
        """Benchmark model inference time"""
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(example_input)

        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(example_input)
                torch.cuda.synchronize() if torch.cuda.is_available() else None

        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        throughput = num_runs / (end_time - start_time)

        return {
            'avg_inference_time': avg_time * 1000,  # ms
            'throughput': throughput,  # inferences per second
            'num_runs': num_runs
        }
```

---

## Real-world Project Implementations

### Image Classification Pipeline

```python
class ImageClassificationPipeline:
    """Complete image classification pipeline with advanced techniques"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize components
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None

    def setup_data(self, data_path, batch_size=32, num_workers=4):
        """Setup data loaders with advanced augmentation"""

        # Data augmentation for training
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                  saturation=0.2, hue=0.1),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.RandomErasing(p=0.1)
        ])

        # Standard validation/test transforms
        val_test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # Create datasets
        train_dataset = datasets.ImageFolder(
            os.path.join(data_path, 'train'),
            transform=train_transform
        )
        val_dataset = datasets.ImageFolder(
            os.path.join(data_path, 'val'),
            transform=val_test_transform
        )
        test_dataset = datasets.ImageFolder(
            os.path.join(data_path, 'test'),
            transform=val_test_transform
        )

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    def setup_model(self, model_name='resnet50', num_classes=1000, pretrained=True):
        """Setup model with advanced techniques"""

        if model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            # Add advanced regularization
            model.avgpool = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(2048, num_classes)
            )
        elif model_name == 'efficientnet_b3':
            model = models.efficientnet_b3(pretrained=pretrained)
            model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(1536, num_classes)
            )
        elif model_name == 'vit_b_16':
            model = models.vit_b_16(pretrained=pretrained)
            model.heads.head = nn.Linear(768, num_classes)

        self.model = model.to(self.device)
        return self.model

    def setup_optimization(self, learning_rate=0.001, weight_decay=1e-4):
        """Setup optimizer, scheduler, and mixed precision training"""

        # Discriminative learning rates
        param_groups = [
            {'params': list(self.model.parameters())[-2:], 'lr': learning_rate * 10},  # Final layers
            {'params': list(self.model.parameters())[:-2], 'lr': learning_rate}  # Backbone
        ]

        self.optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)

        # Cosine annealing with warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=learning_rate * 0.01
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Mixed precision scaler
        self.scaler = GradScaler()

        return self.optimizer, self.scheduler, self.criterion

    def train(self, num_epochs=100, early_stopping_patience=10):
        """Advanced training loop with all optimizations"""

        # Training diagnostics
        diagnostics = TrainingDiagnostics(self.model, self.device)

        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_acc = self.train_epoch()

            # Validation phase
            val_loss, val_acc = self.validate()

            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Track metrics
            gradient_norm = self.get_gradient_norm()
            diagnostics.track_training_metrics(
                train_loss, train_acc, val_loss, val_acc, current_lr, gradient_norm
            )

            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}, Gradient Norm: {gradient_norm:.4f}")
            print("-" * 60)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break

        # Plot training curves
        diagnostics.plot_training_curves()

        # Analyze gradient flow
        gradient_problems = diagnostics.identify_gradient_problems()
        if gradient_problems:
            print("Gradient Flow Issues Detected:")
            for problem in gradient_problems:
                print(f"- {problem['type']}: {problem['description']}")

    def train_epoch(self):
        """Single training epoch with mixed precision"""

        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Mixed precision forward pass
            with autocast():
                output = self.model(data)
                loss = self.criterion(output, target)

            # Scale and backward pass
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # Update running statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')

        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate(self):
        """Validation loop"""

        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)

                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)

                val_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = val_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def get_gradient_norm(self):
        """Compute gradient norm for monitoring"""
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def evaluate_final_model(self):
        """Comprehensive model evaluation"""

        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.model.eval()

        # Get predictions
        analyzer = ModelAnalyzer(self.model, self.test_loader, self.device)
        labels, preds, confidences = analyzer.evaluate_model()

        # Get class names
        class_names = self.test_loader.dataset.classes

        # Generate comprehensive report
        class_metrics = analyzer.analyze_per_class_performance(class_names)

        # Save results
        class_metrics.to_csv('per_class_metrics.csv', index=False)

        # Plot confusion matrix
        analyzer.plot_confusion_matrix(class_names)

        # Analyze confidence distribution
        confidence_stats = analyzer.analyze_confidence_distribution()

        # Print final results
        print("\nFinal Model Evaluation:")
        print("=" * 50)
        print(f"Overall Test Accuracy: {np.mean(labels == preds) * 100:.2f}%")
        print(f"Mean Confidence: {np.mean(np.max(confidences, axis=1)):.4f}")
        print(f"Confidence Std: {np.std(np.max(confidences, axis=1)):.4f}")

        # Top performing classes
        top_classes = class_metrics.nlargest(5, 'F1-Score')[['Class', 'F1-Score']]
        print("\nTop 5 Performing Classes:")
        for _, row in top_classes.iterrows():
            print(f"{row['Class']}: {row['F1-Score']:.4f}")

        # Worst performing classes
        bottom_classes = class_metrics.nsmallest(5, 'F1-Score')[['Class', 'F1-Score']]
        print("\nBottom 5 Performing Classes:")
        for _, row in bottom_classes.iterrows():
            print(f"{row['Class']}: {row['F1-Score']:.4f}")

# Usage example
def main():
    # Configuration
    config = {
        'data_path': './data/imagenet',
        'model_name': 'resnet50',
        'num_classes': 1000,
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 100
    }

    # Initialize pipeline
    pipeline = ImageClassificationPipeline(config)

    # Setup components
    pipeline.setup_data(config['data_path'], config['batch_size'])
    pipeline.setup_model(config['model_name'], config['num_classes'])
    pipeline.setup_optimization(config['learning_rate'])

    # Train and evaluate
    pipeline.train(config['num_epochs'])
    pipeline.evaluate_final_model()

# if __name__ == "__main__":
#     main()
```

This comprehensive practice guide provides real-world implementations of advanced deep learning concepts, including:

1. **ResNet from scratch** with full training pipeline
2. **Transfer learning** with progressive unfreezing
3. **Advanced regularization** including DropBlock, scheduled dropout, and custom batch norm variants
4. **Model evaluation and debugging** with comprehensive diagnostic tools
5. **Hyperparameter tuning** using Bayesian optimization and other search strategies
6. **Ensemble methods** with different voting strategies and snapshot ensembles
7. **Performance optimization** with mixed precision, gradient accumulation, and memory management
8. **Complete production pipeline** with all advanced techniques integrated

Each implementation includes detailed code, explanations, and best practices for production deployment.
