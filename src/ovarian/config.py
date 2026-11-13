"""Shared constants and the hyper-parameters of the three reported models."""

CLASS_NAMES = ["Endometrioma", "Normal_ovary", "PCO", "Serous_cystadenoma", "Teratoma"]
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Augmentation presets (RandomAffine parameters)
AUGMENTATION = {
    "none":     dict(degrees=0,  translate=None,        scale=(1.0, 1.0),   shear=0),
    "mild":     dict(degrees=5,  translate=(0.02, 0.02), scale=(0.95, 1.05), shear=2),
    "moderate": dict(degrees=10, translate=(0.05, 0.05), scale=(0.90, 1.10), shear=5),
    "strong":   dict(degrees=20, translate=(0.10, 0.10), scale=(0.80, 1.20), shear=10),
}

# --- Reported configurations (values taken from the project's tuning records) -----------
CNN_CONFIG = dict(            # "Trial 12" of the CNN search
    architecture="deeper",    # 6 conv blocks: 32-32-64-64-128-128, BatchNorm, global avg pool
    learning_rate=0.000305,
    dropout=0.460104,
    weight_decay=0.001,
    batch_size=32,
    augmentation="mild",
    label_smoothing=0.02,
)

RESNET_CONFIG = dict(         # best Optuna trial (#19) of the ResNet18 search
    learning_rate=0.00028694,
    weight_decay=0.0062081,
    batch_size=32,
    augmentation="moderate",
    label_smoothing=0.05,
)

SVM_CONFIG = dict(C=30, gamma=0.001, kernel="rbf")   # no class weights

# --- Training protocol shared by CNN and ResNet18 ---------------------------------------
TRAIN_PROTOCOL = dict(
    optimizer="AdamW",
    max_epochs=100,
    early_stopping_patience=15,      # on validation macro-F1
    min_delta=0.0005,
    scheduler="ReduceLROnPlateau(mode=max, factor=0.5, patience=4, min_lr=1e-7)",
    mixed_precision=True,            # only used when CUDA is available
    n_folds=5,
    seed=42,
)
