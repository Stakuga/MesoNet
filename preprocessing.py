import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import random

class FacePreprocessor:
    def __init__(self, target_size=(256, 256), confidence_threshold=0.7):
        # Initialize BlazeFace
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=confidence_threshold
        )
        self.target_size = target_size
    
    def detect_and_crop_face(self, image):
        """Detect face using BlazeFace and crop to 256x256"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        if results.detections:
            # Get the most confident detection
            best_detection = max(results.detections, key=lambda x: x.score[0])
            bbox = best_detection.location_data.relative_bounding_box
            
            h, w, _ = image.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Add padding
            padding_x = int(width * 0.1)
            padding_y = int(height * 0.1)
            
            x = max(0, x - padding_x)
            y = max(0, y - padding_y)
            x2 = min(w, x + width + 2 * padding_x)
            y2 = min(h, y + height + 2 * padding_y)
            
            # Crop face
            face = image[y:y2, x:x2]
            
            if face.size > 0:
                face_resized = cv2.resize(face, self.target_size)
                return face_resized
        
        return None

    def extract_random_frame_from_video(self, video_path):
        """Extract one random frame from an MP4 video"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return None
        
        # Get total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            return None
        
        # Choose a random frame number
        random_frame = random.randint(0, total_frames - 1)
        
        # Seek to the random frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
        
        # Read the frame
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return frame
        else:
            return None

class DatasetProcessor:
    def __init__(self, face_preprocessor):
        self.face_preprocessor = face_preprocessor
    
    def process_dataset(self, raw_data_dir, processed_data_dir, train_ratio=0.8):
        """
        Process raw dataset and create train/test split
        
        Expected structure:
        raw_data_dir/
        ├── real/
        │   ├── video1.mp4
        │   ├── video2.mp4
        │   └── ...
        └── fake/
            ├── video1.mp4
            ├── video2.mp4
            └── ...
        """
        raw_path = Path(raw_data_dir)
        processed_path = Path(processed_data_dir)
        
        # Create output directories
        train_dir = processed_path / 'train'
        test_dir = processed_path / 'test'
        
        for class_name in ['real', 'fake']:
            (train_dir / class_name).mkdir(parents=True, exist_ok=True)
            (test_dir / class_name).mkdir(parents=True, exist_ok=True)
        
        print("🔄 Processing video dataset...")
        
        # Process each class
        total_processed = 0
        for class_name in ['real', 'fake']:
            class_dir = raw_path / class_name
            if not class_dir.exists():
                print(f"⚠️  {class_dir} not found, skipping {class_name}")
                continue
            
            # Get all MP4 video files
            video_files = list(class_dir.glob('*.mp4')) + list(class_dir.glob('*.MP4'))
            
            if len(video_files) == 0:
                print(f"⚠️  No MP4 videos found in {class_dir}")
                continue
            
            # Shuffle and split videos
            random.shuffle(video_files)
            split_idx = int(len(video_files) * train_ratio)
            train_videos = video_files[:split_idx]
            test_videos = video_files[split_idx:]
            
            print(f"\n📁 {class_name.upper()}: {len(video_files)} total videos")
            print(f"   Train videos: {len(train_videos)}, Test videos: {len(test_videos)}")
            
            # Process train videos
            train_success = 0
            for video_file in tqdm(train_videos, desc=f"Processing {class_name} train videos"):
                # Extract random frame from video
                frame = self.face_preprocessor.extract_random_frame_from_video(video_file)
                
                if frame is not None:
                    # Detect and crop face from the frame
                    face = self.face_preprocessor.detect_and_crop_face(frame)
                    
                    if face is not None:
                        # Save the face image
                        output_path = train_dir / class_name / f"{video_file.stem}_face.jpg"
                        cv2.imwrite(str(output_path), face)
                        train_success += 1
            
            # Process test videos
            test_success = 0
            for video_file in tqdm(test_videos, desc=f"Processing {class_name} test videos"):
                # Extract random frame from video
                frame = self.face_preprocessor.extract_random_frame_from_video(video_file)
                
                if frame is not None:
                    # Detect and crop face from the frame
                    face = self.face_preprocessor.detect_and_crop_face(frame)
                    
                    if face is not None:
                        # Save the face image
                        output_path = test_dir / class_name / f"{video_file.stem}_face.jpg"
                        cv2.imwrite(str(output_path), face)
                        test_success += 1
            
            class_total = train_success + test_success
            total_processed += class_total
            print(f"   ✅ {class_name}: {class_total} faces extracted from {len(video_files)} videos")
        
        print(f"\n✅ Dataset processing complete!")
        print(f"   Total faces extracted: {total_processed}")
        print(f"   Saved to: {processed_data_dir}")
        
        return total_processed

def main():
    """Example usage of video preprocessing pipeline"""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Initialize preprocessor
    face_preprocessor = FacePreprocessor(
        target_size=(256, 256),
        confidence_threshold=0.7
    )
    
    # Initialize dataset processor
    processor = DatasetProcessor(face_preprocessor)
    
    # Process dataset
    processor.process_dataset(
        raw_data_dir="raw_videos",  # Your raw video folder with real/ and fake/ subfolders
        processed_data_dir="processed_faces",
        train_ratio=0.8
    )

if __name__ == "__main__":
    main()