file_names = ["eth", "hotel", "univ-001", "univ-003", "univ-example", "zara1", "zara2", "zara3"]
train_val_reference = {
    "eth": {"val": [0], "train": [1, 2, 3, 4, 5, 6, 7]},
    "hotel": {"val": [1], "train": [0, 2, 3, 4, 5, 6, 7]},
    "univ": {"val": [2, 3], "train": [0, 1, 4, 5, 6, 7]},
    "zara1": {"val": [5], "train": [0, 1, 2, 3, 4, 6, 7]},
    "zara2": {"val": [6], "train": [0, 1, 2, 3, 4, 5, 7]},
}

scene_range_reference = {  # Manually annotated scene ranges, format ((x_min, x_max), (y_min, y_max))
    "eth": ((-8.5, 16), (-9.9, 21.5)),
    "hotel": ((-6.5, 5.9), (-10.4, 4.8)),
    "univ-001": ((-0.6, 15.5), (-0.3, 14)),
    "univ-003": ((-0.6, 15.5), (-0.3, 14)),
    "univ-example": ((-0.6, 15.7), (-0.8, 14)),
    "zara1": ((-0.3, 15.5), (-0.7, 14.2)),
    "zara2": ((-0.4, 15.6), (-0.3, 16.7)),
    "zara3": ((-0.4, 15.6), (-0.3, 16.7))
}
