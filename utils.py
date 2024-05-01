import torch
from typing import Any, Dict, List, Tuple
from torch.utils.data import default_collate

def custom_collate(batch: List[Tuple[List[torch.Tensor], List[List[torch.Tensor]]]]) -> Dict[str, Any]:
        """
        Custom collate function to handle batches of crops and bounding boxes.

        Args:
            batch (List[Tuple[List[torch.Tensor], List[torch.Tensor]]]): A list of tuples, where each tuple contains:
                - A list of image tensors (crops)
                - A list of tensors of bounding boxes

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'images': a tensor containing all the crops stacked together.
                - 'boxes': a list of tensors, each containing bounding boxes for the corresponding image crop.
        """
        # Unpack the batch and separate crops and boxes
        all_crops = [item for tuple in batch for item in tuple[0]]  # Flatten list of crop lists
        all_boxes = [list for tuple in batch for list in tuple[1]]  # Flatten list of lists of box lists

        # Use default_collate to handle images which appropriately stacks them into a single tensor
        images_collated = default_collate(all_crops)

        # Bounding boxes don't need to be stacked into a single tensor since each box is associated with a crop
        # However, they need to be kept in lists corresponding to each image crop
        boxes_collated = all_boxes

        return {'images': images_collated, 'boxes': boxes_collated}

