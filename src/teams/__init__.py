"""
src.teams — Team classification modules.

Modules
-------
crop_extractor  : Crop the torso region from each player bounding box.
colour_features : Extract colour features (mean, histogram, K-Means centroids)
                  from player crops in RGB, HSV, or LAB colour spaces.
classifier      : Cluster colour features to assign Team A / Team B / Unknown labels
                  and stabilize them across frames using majority voting.
auto_classifier : Automatic K-Means kit color discovery (sequence-agnostic).
role_refiner    : Spatial & temporal post-processing engine that corrects
                  goalkeeper, staff, and team labels using pitch geometry.
"""
