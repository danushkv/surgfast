import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List
import cv2
from PIL import Image
from torch.utils.data import Dataset


class CholecTrainDataset(Dataset):
    """
    PyTorch Dataset for training medsiglip model with cholecystectomy images.
    Returns images with bounding boxes drawn and captions generated from labels.
    Handles hierarchical directory structure with multiple video folders.
    
    Strategy:
    - Uses all folders from 'train' directory
    - Splits 'val' folders 50/50: 9 folders for training, 9 for validation
    - Can save folder split information to a text file
    """
    
    def __init__(self, dataset_root: str, split_type: str = 'train', transform=None,
                 seed: int = 42):
        """
        Initialize the training dataset.
        
        Args:
            dataset_root: Path to root dataset directory (e.g., 'dataset/cholecinstseg')
            split_type: 'train' (all train folders + 9 val folders) or 
                       'val' (remaining 9 val folders)
            transform: Optional transform to apply to images
            seed: Random seed for reproducible val folder splitting
        """
        self.dataset_root = Path(dataset_root)
        self.split_type = split_type
        self.transform = transform
        self.seed = seed
        
        # Determine folders to use
        self.selected_folders = self._select_folders()
        
        # Find all annotation files
        self.ann_files = self._find_all_annotations()
        
        if not self.ann_files:
            raise ValueError(f"No annotation files found for {split_type} split")
        
        print(f"Found {len(self.ann_files)} annotation files for {split_type} split")
        print(f"Using folders: {sorted(self.selected_folders)}")
        
        # Color map for surgical instruments with their display names
        self.color_map = {
            'grasper': (0, 255, 0),
            'bipolar': (255, 0, 0),
            'hook': (0, 0, 255),
            'scissors': (255, 255, 0),
            'clipper': (255, 0, 255),
            'specimen': (0, 255, 255),
        }
        
        self.color_names = {
            (0, 255, 0): 'green',
            (255, 0, 0): 'blue',
            (0, 0, 255): 'red',
            (255, 255, 0): 'cyan',
            (255, 0, 255): 'magenta',
            (0, 255, 255): 'yellow',
        }
    
    def _select_folders(self) -> set:
        """
        Select folders based on split type.
        - 'train': all train folders + first half of val folders
        - 'val': remaining half of val folders only
        
        Returns:
            Set of selected folder names
        """
        selected = set()
        
        # Add all train folders ONLY if split_type is 'train'
        if self.split_type == 'train':
            train_path = self.dataset_root / 'train'
            if train_path.exists():
                for folder_path in train_path.iterdir():
                    if folder_path.is_dir() and (folder_path / 'ann_dir').exists():
                        selected.add(folder_path.name)
        
        # Split val folders
        val_path = self.dataset_root / 'val'
        if val_path.exists():
            val_folders = []
            for folder_path in val_path.iterdir():
                if folder_path.is_dir() and (folder_path / 'ann_dir').exists():
                    val_folders.append(folder_path.name)
            
            val_folders.sort()
            shuffled = val_folders.copy()
            random.Random(self.seed).shuffle(shuffled)
            
            train_val_folders = shuffled[:len(shuffled) // 2]
            test_val_folders = shuffled[len(shuffled) // 2:]
            
            # Store for later reference
            self.train_val_folders = sorted(train_val_folders)
            self.test_val_folders = sorted(test_val_folders)
            
            if self.split_type == 'train':
                selected.update(self.train_val_folders)
            elif self.split_type == 'val':
                selected.update(self.test_val_folders)
        
        return selected
    
    def _find_all_annotations(self) -> List[Path]:
        """
        Recursively find all JSON annotation files in selected folders.
        
        Returns:
            Sorted list of Path objects pointing to JSON annotation files
        """
        ann_files = []
        
        # Find annotations in both train and val directories
        for base_split in ['train', 'val']:
            base_path = self.dataset_root / base_split
            if not base_path.exists():
                continue
            
            for folder in self.selected_folders:
                folder_path = base_path / folder / 'ann_dir'
                if folder_path.exists():
                    ann_files.extend(sorted(folder_path.glob('*.json')))
        
        return sorted(ann_files)
    
    def _get_image_path(self, ann_file: Path) -> Path:
        """
        Get the corresponding image path for an annotation file.
        
        Assumes structure: .../ann_dir/annotation.json -> .../img_dir/image.png
        
        Args:
            ann_file: Path to annotation JSON file
            
        Returns:
            Path to corresponding image file
        """
        # Replace 'ann_dir' with 'img_dir' in the path
        img_dir = ann_file.parent.parent / 'img_dir'
        
        # Get the JSON filename and replace extension with .png
        img_name = ann_file.stem + '.png'
        img_path = img_dir / img_name
        
        return img_path
    
    def __len__(self):
        return len(self.ann_files)
    
    def _generate_caption(self, bboxes: List[Dict]) -> str:
        """
        Generate caption from bounding box labels.
        
        Args:
            bboxes: List of bounding box dictionaries
            
        Returns:
            Caption string describing the image content
        """
        if not bboxes:
            return "cholecystectomy surgical image"
        
        if len(bboxes) == 1:
            label = bboxes[0]['label']
            return f"cholecystectomy image with {label}"
        
        # Multiple labels - include color information
        label_color_pairs = []
        for bbox in bboxes:
            label = bbox['label']
            color = self.color_map.get(label, (255, 255, 255))
            color_name = self.color_names.get(color, 'white')
            label_color_pairs.append(f"{label} shown in {color_name}")
        
        labels_text = " and ".join(label_color_pairs)
        return f"cholecystectomy image with {labels_text}"
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get an item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing:
                - 'image': PIL Image with bounding boxes drawn
                - 'caption': String caption describing the image
                - 'pixel_values': Transformed image tensor (if transform provided)
        """
        ann_file = self.ann_files[idx]
        
        # Load annotation from JSON
        with open(ann_file, 'r') as f:
            annotation = json.load(f)
        
        # Get image path
        img_path = self._get_image_path(ann_file)
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Load image
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get bounding boxes from annotation
        bboxes = []
        if 'shapes' in annotation and annotation['shapes']:
            for shape in annotation['shapes']:
                points = np.array(shape['points'], dtype=np.float32)
                label = shape.get('label', 'unknown')
                
                # Calculate bounding box
                x_min = int(np.min(points[:, 0]))
                y_min = int(np.min(points[:, 1]))
                x_max = int(np.max(points[:, 0]))
                y_max = int(np.max(points[:, 1]))
                
                bboxes.append({
                    'label': label,
                    'bbox': [x_min, y_min, x_max, y_max],
                })
        
        # Draw bounding boxes on image
        for bbox_info in bboxes:
            label = bbox_info['label']
            x_min, y_min, x_max, y_max = bbox_info['bbox']
            color = self.color_map.get(label, (255, 255, 255))
            
            # OpenCV uses BGR, convert RGB color to BGR
            color_bgr = (color[2], color[1], color[0])
            cv2.rectangle(img_rgb, (x_min, y_min), (x_max, y_max), color_bgr, 2)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(img_rgb)
        
        # Generate caption
        caption = self._generate_caption(bboxes)
        
        result = {
            'image': pil_img,
            'caption': caption,
        }
        
        # Apply transform if provided
        if self.transform:
            result['pixel_values'] = self.transform(pil_img)
        
        return result
    
    def save_split_info(self, output_path: str) -> None:
        """
        Save the train/val split information to a text file.
        
        Args:
            output_path: Path where to save the split information file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("CHOLECYSTECTOMY DATASET SPLIT INFORMATION\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Seed: {self.seed}\n")
            f.write(f"Dataset Root: {self.dataset_root}\n\n")
            
            # Get train folders
            train_path = self.dataset_root / 'train'
            train_folders = []
            if train_path.exists():
                for folder_path in sorted(train_path.iterdir()):
                    if folder_path.is_dir() and (folder_path / 'ann_dir').exists():
                        train_folders.append(folder_path.name)
            
            f.write("TRAINING SET FOLDERS:\n")
            f.write("-" * 60 + "\n")
            f.write("From train/:\n")
            for folder in sorted(train_folders):
                f.write(f"  - {folder}\n")
            
            if hasattr(self, 'train_val_folders'):
                f.write("\nFrom val/ (added to training):\n")
                for folder in sorted(self.train_val_folders):
                    f.write(f"  - {folder}\n")
            
            f.write(f"\nTotal training folders: {len(train_folders) + len(getattr(self, 'train_val_folders', []))}\n")
            f.write(f"Total training images: {len(self.ann_files)}\n\n")
            
            # Get val folders
            f.write("VALIDATION SET FOLDERS:\n")
            f.write("-" * 60 + "\n")
            if hasattr(self, 'test_val_folders'):
                for folder in sorted(self.test_val_folders):
                    f.write(f"  - {folder}\n")
                f.write(f"\nTotal validation folders: {len(self.test_val_folders)}\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        print(f"Split information saved to {output_file}")


class CholecMedGemmaDataset(Dataset):
    """
    PyTorch Dataset for training MedGemma model on cholecystectomy images.
    Generates chat-style messages with tool detection prompts.
    """
    
    def __init__(self, dataset_root: str, split_type: str = 'train', seed: int = 42):
        """
        Initialize the MedGemma training dataset.
        
        Args:
            dataset_root: Path to root dataset directory (e.g., 'dataset/cholecinstseg')
            split_type: 'train' (all train folders + 9 val folders) or 'val' (remaining 9 val folders)
            seed: Random seed for reproducible val folder splitting
        """
        self.processor = CholecTrainDataset(
            dataset_root=dataset_root,
            split_type=split_type,
            seed=seed
        )
        self.ann_files = self.processor.ann_files
        self.dataset_root = Path(dataset_root)
        self.split_type = split_type
        self.seed = seed
        
        # Tool name mappings for readable names
        self.tool_names = {
            'grasper': 'Grasper',
            'bipolar': 'Bipolar',
            'hook': 'Hook',
            'scissors': 'Scissors',
            'clipper': 'Clipper',
            'specimen': 'Specimen',
        }
    
    def __len__(self):
        return len(self.ann_files)
    
    def _get_image_path(self, ann_file: Path) -> Path:
        """
        Get the corresponding image path for an annotation file.
        
        Args:
            ann_file: Path to annotation JSON file
            
        Returns:
            Path to corresponding image file
        """
        # Replace 'ann_dir' with 'img_dir' in the path
        img_dir = ann_file.parent.parent / 'img_dir'
        img_name = ann_file.stem + '.png'
        img_path = img_dir / img_name
        return img_path
    
    def _generate_answer(self, bboxes: List[Dict]) -> str:
        """
        Generate answer text from bounding boxes.
        
        Args:
            bboxes: List of bounding box dictionaries
            
        Returns:
            Answer describing tools present in the image
        """
        if not bboxes:
            return "No surgical tools are visible in this image."
        
        tool_list = [self.tool_names.get(bbox['label'], bbox['label']) for bbox in bboxes]
        
        if len(tool_list) == 1:
            return f"The tool present in the cholecystectomy image is: {tool_list[0]}."
        else:
            tools_text = ", ".join(tool_list[:-1]) + f", and {tool_list[-1]}"
            return f"The tools present in the cholecystectomy image are: {tools_text}."
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get an item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing 'image' and 'messages' (chat format)
        """
        ann_file = self.ann_files[idx]
        
        # Load annotation
        with open(ann_file, 'r') as f:
            annotation = json.load(f)
        
        # Get image path and load image
        img_path = self._get_image_path(ann_file)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = Image.open(str(img_path)).convert("RGB")
        
        # Get bounding boxes from annotation
        bboxes = []
        if 'shapes' in annotation and annotation['shapes']:
            for shape in annotation['shapes']:
                points = np.array(shape['points'], dtype=np.float32)
                label = shape.get('label', 'unknown')
                
                # Calculate bounding box
                x_min = int(np.min(points[:, 0]))
                y_min = int(np.min(points[:, 1]))
                x_max = int(np.max(points[:, 0]))
                y_max = int(np.max(points[:, 1]))
                
                bboxes.append({
                    'label': label,
                    'bbox': [x_min, y_min, x_max, y_max],
                })
        
        # Generate chat-style messages
        answer = self._generate_answer(bboxes)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {
                        "type": "text",
                        "text": "What are the tools present in the cholecystectomy image?",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": answer,
                    },
                ],
            },
        ]
        
        return {
            "image": image,
            "messages": messages,
        }
