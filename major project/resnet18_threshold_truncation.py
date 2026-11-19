# ================================================================
# RESNET-18 RANDOM PARAMETER SELECTION EXPERIMENT
#
# CIFAR-100: Training
# MiniImageNet: Evaluation only
#
# Independently select 10% parameters from:
# Layer 1, Layer 2, Layer 3, Layer 4
#
# MiniImageNet evaluation:
# 1. Classification Accuracy
# 2. KL Divergence
# 3. Prediction Agreement
# ================================================================


import os
import copy
import json
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights

from tqdm.auto import tqdm


# ================================================================
# 1. CONFIGURATION
# ================================================================

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CIFAR_PATH = (
    "/kaggle/input/datasets/melikechan/cifar100/cifar100"
)

MINI_PATH = (
    "/kaggle/input/datasets/deeptrial/"
    "miniimagenet/ImageNet-Mini/images"
)

IMAGENET_JSON = (
    "/kaggle/input/datasets/deeptrial/"
    "miniimagenet/ImageNet-Mini/imagenet_class_index.json"
)

OUTPUT_DIR = "/kaggle/working/resnet18_experiment"

os.makedirs(OUTPUT_DIR, exist_ok=True)


BATCH_SIZE = 64

IMAGE_SIZE = 224

EPOCHS_PER_ITERATION = 5

MAX_ITERATIONS = 40

SELECTION_PERCENTAGE = 0.10\n\n# Threshold uses mean(abs(weight)) separately for each ResNet stage.\n# Only selected weights with abs(weight) < threshold are updated.\nOLD_WEIGHT_FACTOR = 0.90\nNEW_WEIGHT_FACTOR = 0.10

PATIENCE = 3

BACKBONE_LR = 0.0001

CLASSIFIER_LR = 0.01

NUM_WORKERS = 2

USE_AMP = DEVICE.type == "cuda"


# Reproducibility

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


print("=" * 70)
print("RESNET-18 RANDOM PARAMETER SELECTION EXPERIMENT")
print("=" * 70)

print("Device:", DEVICE)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Selection:", SELECTION_PERCENTAGE * 100, "% per stage")
print("Epochs per iteration:", EPOCHS_PER_ITERATION)
print("Maximum iterations:", MAX_ITERATIONS)


# ================================================================
# 2. TRANSFORMS
# ================================================================

train_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


test_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ================================================================
# 3. LOAD CIFAR-100
# ================================================================

train_dir = os.path.join(CIFAR_PATH, "train")

test_dir = os.path.join(CIFAR_PATH, "test")


if not os.path.isdir(train_dir):
    raise FileNotFoundError(train_dir)

if not os.path.isdir(test_dir):
    raise FileNotFoundError(test_dir)


train_dataset = datasets.ImageFolder(
    train_dir,
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=test_transform
)


if train_dataset.class_to_idx != test_dataset.class_to_idx:
    raise ValueError("CIFAR train/test class mappings differ.")


if len(train_dataset.classes) != 100:
    raise ValueError("Expected 100 CIFAR classes.")


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0)
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0)
)


print("\nCIFAR-100 loaded successfully")

print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))
print("Classes:", len(train_dataset.classes))


# ================================================================
# 4. LOAD IMAGENET CLASS INDEX
# ================================================================

with open(IMAGENET_JSON, "r") as f:
    imagenet_class_index = json.load(f)


# Convert:
# n01440764 -> 0
# n01443537 -> 1
# etc.

wnid_to_idx = {

    value[0]: int(key)

    for key, value in imagenet_class_index.items()

}


print("\nImageNet class mapping loaded")

print("ImageNet classes:", len(wnid_to_idx))


# ================================================================
# 5. MINIIMAGENET LABEL MAPPER
# ================================================================

class ImageNetLabelMapper:

    def __init__(self, folder_classes, mapping):

        missing = [
            name for name in folder_classes
            if name not in mapping
        ]

        if missing:

            raise ValueError(
                f"Missing ImageNet mappings: {missing[:10]}"
            )

        self.class_indices = [

            mapping[class_name]

            for class_name in folder_classes

        ]


    def __call__(self, folder_target):

        return self.class_indices[folder_target]


# ================================================================
# 6. LOAD MINIIMAGENET
# ================================================================

if not os.path.isdir(MINI_PATH):
    raise FileNotFoundError(MINI_PATH)


# First create dataset to get folder class ordering.

mini_dataset = datasets.ImageFolder(
    MINI_PATH,
    transform=test_transform
)


# Verify class mapping.

if len(mini_dataset.classes) != 1000:
    raise ValueError(
        f"Expected 1000 classes, found {len(mini_dataset.classes)}"
    )


# Apply correct ImageNet labels.

mini_dataset.target_transform = ImageNetLabelMapper(
    mini_dataset.classes,
    wnid_to_idx
)


mini_loader = DataLoader(
    mini_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0)
)


print("\nMiniImageNet loaded successfully")

print("Images:", len(mini_dataset))
print("Classes:", len(mini_dataset.classes))

print("ImageNet label mapping verified.")


# ================================================================
# 7. LOAD PRETRAINED RESNET-18
# ================================================================

print("\nLoading pretrained ImageNet ResNet-18...")


original_model = models.resnet18(
    weights=ResNet18_Weights.DEFAULT
).to(DEVICE)


original_model.eval()


for param in original_model.parameters():
    param.requires_grad = False


# ================================================================
# 8. CREATE STUDENT MODEL
# ================================================================

class ResNet18CIFAR(nn.Module):

    def __init__(self, pretrained_model):

        super().__init__()

        # Copy pretrained ResNet-18

        self.backbone = copy.deepcopy(pretrained_model)

        # Preserve original ImageNet classifier

        self.imagenet_probe = copy.deepcopy(
            pretrained_model.fc
        )

        # Remove FC from backbone

        self.backbone.fc = nn.Identity()

        # New CIFAR-100 classifier

        self.cifar_classifier = nn.Linear(
            512,
            100
        )

        # Freeze all backbone parameters initially.

        for param in self.backbone.parameters():
            param.requires_grad = False

        # Freeze ImageNet evaluation classifier.

        for param in self.imagenet_probe.parameters():
            param.requires_grad = False

        # Enable Layer 1, 2, 3, 4.

        for stage in [

            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4

        ]:

            for param in stage.parameters():
                param.requires_grad = True

        # CIFAR classifier fully trainable.

        for param in self.cifar_classifier.parameters():
            param.requires_grad = True


    def forward(self, x):

        features = self.backbone(x)

        return self.cifar_classifier(features)


    def imagenet_forward(self, x):

        features = self.backbone(x)

        return self.imagenet_probe(features)


student_model = ResNet18CIFAR(
    original_model
).to(DEVICE)


print("\nStudent model created successfully.")


# ================================================================
# 9. PARAMETER COUNTS
# ================================================================

stages = {

    "Layer 1": student_model.backbone.layer1,

    "Layer 2": student_model.backbone.layer2,

    "Layer 3": student_model.backbone.layer3,

    "Layer 4": student_model.backbone.layer4

}


print("\n" + "=" * 70)
print("PARAMETER DISTRIBUTION")
print("=" * 70)


for name, stage in stages.items():

    count = sum(
        p.numel()
        for p in stage.parameters()
    )

    print(
        f"{name}: {count:,} parameters | "
        f"10% = {int(count * 0.10):,}"
    )


# ================================================================
# 10. RANDOM 10% SELECTION PER STAGE
# ================================================================

current_masks = {}


def generate_stage_masks(model):

    global current_masks

    current_masks = {}

    selected_counts = {}


    for stage_name in [

        "layer1",
        "layer2",
        "layer3",
        "layer4"

    ]:

        stage = getattr(
            model.backbone,
            stage_name
        )

        parameters = list(
            stage.named_parameters()
        )

        total_parameters = sum(
            param.numel()
            for _, param in parameters
        )

        number_selected = max(
            1,
            int(total_parameters * SELECTION_PERCENTAGE)
        )


        # Generate an independent random selection
        # for this stage.

        random_indices = torch.randperm(
            total_parameters,
            device=DEVICE
        )[:number_selected]


        random_indices = torch.sort(
            random_indices
        )[0]


        offset = 0


        for parameter_name, param in parameters:

            number = param.numel()

            local_indices = random_indices[

                (random_indices >= offset) &
                (random_indices < offset + number)

            ] - offset


            mask = torch.zeros(
                number,
                device=DEVICE,
                dtype=torch.bool
            )


            if local_indices.numel() > 0:

                mask[local_indices] = True


            mask = mask.reshape(param.shape)


            full_name = (
                f"backbone.{stage_name}.{parameter_name}"
            )

            current_masks[full_name] = mask


            offset += number


        selected_counts[stage_name] = (
            number_selected,
            total_parameters
        )


    print("\nRANDOM PARAMETER SELECTION")


    for name, (selected, total) in selected_counts.items():

        print(
            f"{name}: {selected:,}/{total:,} "
            f"({selected / total * 100:.2f}%)"
        )


    return current_masks


# ================================================================
# 11. THRESHOLD-AWARE GRADIENT MASKS
# ================================================================

# current_masks selects exactly 10% of parameters in each ResNet stage.
# current_update_masks additionally applies the average-absolute-weight
# threshold. Only selected weights with abs(weight) < stage threshold update.

current_update_masks = {}


def build_threshold_update_masks(model):
    global current_update_masks

    current_update_masks = {}
    stage_thresholds = {}

    for stage_name in ["layer1", "layer2", "layer3", "layer4"]:
        stage = getattr(model.backbone, stage_name)

        # Average absolute value across ALL weights in this stage.
        all_abs_weights = torch.cat([
            p.detach().abs().reshape(-1)
            for p in stage.parameters()
        ])
        threshold = all_abs_weights.mean()
        stage_thresholds[stage_name] = threshold.item()

        for parameter_name, param in stage.named_parameters():
            full_name = f"backbone.{stage_name}.{parameter_name}"
            selected_mask = current_masks[full_name]
            below_threshold = param.detach().abs() < threshold
            current_update_masks[full_name] = selected_mask & below_threshold

    print("\nSTAGE AVERAGE-ABSOLUTE-WEIGHT THRESHOLDS")
    for name, value in stage_thresholds.items():
        print(f"{name}: {value:.8f}")

    return current_update_masks


def register_gradient_hooks(model):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not name.startswith("backbone.layer"):
            continue

        def mask_gradient(gradient, parameter_name=name):
            mask = current_update_masks[parameter_name]
            return gradient * mask

        param.register_hook(mask_gradient)


register_gradient_hooks(student_model)


# ================================================================
# 12. OPTIMIZER
# ================================================================

backbone_parameters = []


for stage_name in [

    "layer1",
    "layer2",
    "layer3",
    "layer4"

]:

    stage = getattr(
        student_model.backbone,
        stage_name
    )

    backbone_parameters.extend(
        stage.parameters()
    )


optimizer = optim.SGD(

    [

        {
            "params": backbone_parameters,
            "lr": BACKBONE_LR
        },

        {
            "params": student_model.cifar_classifier.parameters(),
            "lr": CLASSIFIER_LR
        }

    ],

    momentum=0.0,

    weight_decay=0.0

)


scaler = torch.amp.GradScaler(
    "cuda",
    enabled=USE_AMP
)


# ================================================================
# 13. FREEZE BATCH NORMALIZATION STATISTICS
# ================================================================

def freeze_batchnorm_statistics(model):

    for module in model.modules():

        if isinstance(module, nn.BatchNorm2d):

            module.eval()


# ================================================================
# 14. CIFAR-100 EVALUATION
# ================================================================

def evaluate_cifar(model):

    model.eval()

    correct = 0

    total = 0

    total_loss = 0.0


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )


            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=USE_AMP
            ):

                outputs = model(images)

                loss = F.cross_entropy(
                    outputs,
                    labels
                )


            total_loss += (
                loss.item() * labels.size(0)
            )


            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)


    accuracy = 100.0 * correct / total

    average_loss = total_loss / total

    return accuracy, average_loss


# ================================================================
# 15. ORIGINAL MINIIMAGENET PREDICTIONS
# ================================================================

print("\nComputing original MiniImageNet predictions...")


original_mini_logits = []


with torch.no_grad():

    for images, _ in tqdm(mini_loader):

        images = images.to(
            DEVICE,
            non_blocking=True
        )


        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP
        ):

            outputs = original_model(images)


        original_mini_logits.append(
            outputs.float().cpu()
        )


original_mini_logits = torch.cat(
    original_mini_logits,
    dim=0
)


print(
    "Original MiniImageNet logits:",
    original_mini_logits.shape
)


# ================================================================
# 16. MINIIMAGENET EVALUATION
# ================================================================

def evaluate_miniimagenet(model):

    model.eval()

    total_kl = 0.0

    total_agreement = 0

    correct = 0

    total = 0

    start_index = 0


    with torch.no_grad():

        for images, labels in mini_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            batch_size = images.size(0)


            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=USE_AMP
            ):

                updated_logits = model.imagenet_forward(
                    images
                )


            updated_logits = updated_logits.float()


            original_logits = original_mini_logits[

                start_index:start_index + batch_size

            ].to(DEVICE)


            # Original ImageNet probability distribution

            original_probabilities = F.softmax(
                original_logits,
                dim=1
            )


            # Updated ImageNet log probability distribution

            updated_log_probabilities = F.log_softmax(
                updated_logits,
                dim=1
            )


            # KL divergence:
            # KL(Original || Updated)

            kl = F.kl_div(

                updated_log_probabilities,

                original_probabilities,

                reduction="sum"

            )


            total_kl += kl.item()


            # Prediction agreement with original model

            original_predictions = original_logits.argmax(
                dim=1
            )

            updated_predictions = updated_logits.argmax(
                dim=1
            )


            total_agreement += (

                original_predictions == updated_predictions

            ).sum().item()


            # Actual MiniImageNet classification accuracy

            correct += (

                updated_predictions == labels

            ).sum().item()


            total += batch_size

            start_index += batch_size


    average_kl = total_kl / total

    agreement = (
        100.0 * total_agreement / total
    )

    accuracy = (
        100.0 * correct / total
    )


    return average_kl, agreement, accuracy


# ================================================================
# 17. TRAIN ONE EPOCH
# ================================================================

def train_one_epoch(model):

    model.train()
    freeze_batchnorm_statistics(model)

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, leave=False):

        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Build the threshold condition using weights BEFORE this update.
        build_threshold_update_masks(model)

        # Keep a copy of the old weights so the candidate SGD update can
        # be blended as: 0.9 * old_weight + 0.1 * candidate_weight.
        old_weights = {}
        for name, param in model.named_parameters():
            if name in current_update_masks:
                old_weights[name] = param.detach().clone()

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP
        ):
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Blend only the selected weights that passed the threshold test.
        # All other weights remain unchanged by this backbone update.
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in current_update_masks:
                    continue

                mask = current_update_masks[name]
                if mask.any():
                    old = old_weights[name]
                    candidate = param.detach().clone()
                    blended = OLD_WEIGHT_FACTOR * old + NEW_WEIGHT_FACTOR * candidate
                    param.copy_(torch.where(mask, blended, old))

        total_loss += loss.item() * labels.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


# ================================================================
# 18. INITIAL EVALUATION
# ================================================================

print("\n" + "=" * 70)

print("INITIAL MODEL EVALUATION")

print("=" * 70)


initial_cifar_accuracy, initial_cifar_loss = evaluate_cifar(
    student_model
)


initial_mini_kl, initial_mini_agreement, initial_mini_accuracy = (
    evaluate_miniimagenet(student_model)
)


print(
    f"Initial CIFAR Accuracy: "
    f"{initial_cifar_accuracy:.2f}%"
)

print(
    f"Initial CIFAR Loss: "
    f"{initial_cifar_loss:.4f}"
)

print(
    f"Initial MiniImageNet Accuracy: "
    f"{initial_mini_accuracy:.2f}%"
)

print(
    f"Initial MiniImageNet KL: "
    f"{initial_mini_kl:.8f}"
)

print(
    f"Initial MiniImageNet Agreement: "
    f"{initial_mini_agreement:.2f}%"
)


# ================================================================
# 19. MAIN ITERATIVE TRAINING
# ================================================================

best_accuracy = initial_cifar_accuracy

best_model_state = copy.deepcopy(
    student_model.state_dict()
)

patience_counter = 0

history = []


print("\n" + "=" * 70)

print("STARTING ITERATIVE TRAINING")

print("=" * 70)


for iteration in range(
    1,
    MAX_ITERATIONS + 1
):

    print("\n")

    print("=" * 70)

    print(
        f"ITERATION {iteration}/{MAX_ITERATIONS}"
    )

    print("=" * 70)


    # ------------------------------------------------------------
    # STEP 1: FRESH RANDOM 10% SELECTION
    # ------------------------------------------------------------

    generate_stage_masks(student_model)


    # ------------------------------------------------------------
    # STEP 2: TRAIN CIFAR-100 FOR 5 EPOCHS
    # ------------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS_PER_ITERATION + 1
    ):

        train_loss, train_accuracy = train_one_epoch(
            student_model
        )


        print(
            f"Iteration {iteration} | "
            f"Epoch {epoch}/{EPOCHS_PER_ITERATION} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}%"
        )


    # ------------------------------------------------------------
    # STEP 3: CIFAR-100 EVALUATION
    # ------------------------------------------------------------

    cifar_accuracy, cifar_loss = evaluate_cifar(
        student_model
    )


    # ------------------------------------------------------------
    # STEP 4: MINIIMAGENET EVALUATION
    # ------------------------------------------------------------

    mini_kl, mini_agreement, mini_accuracy = (
        evaluate_miniimagenet(student_model)
    )


    # ------------------------------------------------------------
    # STEP 5: DISPLAY ALL RESULTS
    # ------------------------------------------------------------

    print("\nITERATION RESULTS")

    print("-" * 50)

    print(
        f"CIFAR-100 Accuracy: {cifar_accuracy:.2f}%"
    )

    print(
        f"CIFAR-100 Loss: {cifar_loss:.4f}"
    )

    print(
        f"MiniImageNet Accuracy: {mini_accuracy:.2f}%"
    )

    print(
        f"MiniImageNet KL Divergence: {mini_kl:.8f}"
    )

    print(
        f"MiniImageNet Prediction Agreement: "
        f"{mini_agreement:.2f}%"
    )


    # ------------------------------------------------------------
    # STEP 6: SAVE HISTORY
    # ------------------------------------------------------------

    history.append({

        "Iteration": iteration,

        "CIFAR_Accuracy": cifar_accuracy,

        "CIFAR_Loss": cifar_loss,

        "MiniImageNet_Accuracy": mini_accuracy,

        "MiniImageNet_KL": mini_kl,

        "MiniImageNet_Agreement": mini_agreement

    })


    # ------------------------------------------------------------
    # STEP 7: SAVE BEST MODEL
    # ------------------------------------------------------------

    if cifar_accuracy > best_accuracy:

        best_accuracy = cifar_accuracy

        best_model_state = copy.deepcopy(
            student_model.state_dict()
        )

        patience_counter = 0


        torch.save(

            best_model_state,

            os.path.join(
                OUTPUT_DIR,
                "best_model.pth"
            )

        )


        print("\nNew best CIFAR-100 model saved.")


    else:

        patience_counter += 1

        print(
            f"No improvement: "
            f"{patience_counter}/{PATIENCE}"
        )


    # ------------------------------------------------------------
    # STEP 8: SAVE HISTORY AFTER EVERY ITERATION
    # ------------------------------------------------------------

    pd.DataFrame(history).to_csv(

        os.path.join(
            OUTPUT_DIR,
            "training_history.csv"
        ),

        index=False

    )


    # ------------------------------------------------------------
    # STEP 9: EARLY STOPPING
    # ------------------------------------------------------------

    if patience_counter >= PATIENCE:

        print("\nEARLY STOPPING TRIGGERED")

        print(
            f"No CIFAR accuracy improvement for "
            f"{PATIENCE} consecutive iterations."
        )

        break


# ================================================================
# 20. LOAD BEST MODEL
# ================================================================

student_model.load_state_dict(
    best_model_state
)


# ================================================================
# 21. FINAL EVALUATION
# ================================================================

print("\n" + "=" * 70)

print("FINAL MODEL EVALUATION")

print("=" * 70)


final_cifar_accuracy, final_cifar_loss = evaluate_cifar(
    student_model
)


final_mini_kl, final_mini_agreement, final_mini_accuracy = (
    evaluate_miniimagenet(student_model)
)


print("\nFINAL RESULTS")

print("-" * 50)

print(
    f"Initial CIFAR Accuracy: "
    f"{initial_cifar_accuracy:.2f}%"
)

print(
    f"Final CIFAR Accuracy: "
    f"{final_cifar_accuracy:.2f}%"
)

print(
    f"Best CIFAR Accuracy: "
    f"{best_accuracy:.2f}%"
)

print(
    f"Initial MiniImageNet Accuracy: "
    f"{initial_mini_accuracy:.2f}%"
)

print(
    f"Final MiniImageNet Accuracy: "
    f"{final_mini_accuracy:.2f}%"
)

print(
    f"Initial MiniImageNet KL: "
    f"{initial_mini_kl:.8f}"
)

print(
    f"Final MiniImageNet KL: "
    f"{final_mini_kl:.8f}"
)

print(
    f"Initial MiniImageNet Agreement: "
    f"{initial_mini_agreement:.2f}%"
)

print(
    f"Final MiniImageNet Agreement: "
    f"{final_mini_agreement:.2f}%"
)


# ================================================================
# 22. SAVE FINAL MODEL AND RESULTS
# ================================================================

torch.save(

    student_model.state_dict(),

    os.path.join(
        OUTPUT_DIR,
        "final_model.pth"
    )

)


history_df = pd.DataFrame(history)


history_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "final_training_history.csv"
    ),

    index=False

)


# Save final summary

summary = {

    "Initial_CIFAR_Accuracy": initial_cifar_accuracy,

    "Final_CIFAR_Accuracy": final_cifar_accuracy,

    "Best_CIFAR_Accuracy": best_accuracy,

    "Initial_MiniImageNet_Accuracy": initial_mini_accuracy,

    "Final_MiniImageNet_Accuracy": final_mini_accuracy,

    "Initial_MiniImageNet_KL": initial_mini_kl,

    "Final_MiniImageNet_KL": final_mini_kl,

    "Initial_MiniImageNet_Agreement": initial_mini_agreement,

    "Final_MiniImageNet_Agreement": final_mini_agreement

}


pd.DataFrame(
    [summary]
).to_csv(

    os.path.join(
        OUTPUT_DIR,
        "final_summary.csv"
    ),

    index=False

)


print("\n" + "=" * 70)

print("EXPERIMENT COMPLETED")

print("=" * 70)

print("\nOutput directory:", OUTPUT_DIR)

print("\nFiles saved:")

print("1. best_model.pth")

print("2. final_model.pth")

print("3. training_history.csv")

print("4. final_training_history.csv")

print("5. final_summary.csv")