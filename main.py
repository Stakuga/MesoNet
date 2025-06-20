import os
import sys

# Add current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from claude_iii import TwoStageMesoViTPipeline
from preprocessing import FacePreprocessor, DatasetProcessor
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
from pathlib import Path

def run_complete_pipeline():
    """Run the entire pipeline without terminal commands"""
    # Configuration - CHANGE THESE PATHS
    RAW_DATA_DIR = "raw_videos"  # Your folder with real/ and fake/ subfolders
    PROCESSED_DATA_DIR = "processed_faces"
    WEIGHTS_PATH = "weights/Meso4_DF.h5"
    TEST_IMAGE = "test_photo.jpg"  # Optional: single image to test
    
    print("🚀 Starting Complete Deepfake Detection Pipeline")
    print("="*60)
    
    # STEP 1: PREPROCESSING
    # print("\n1️⃣ PREPROCESSING DATASET...")
    
    # if Path(RAW_DATA_DIR).exists():
    #     # Initialize preprocessor
    #     face_preprocessor = FacePreprocessor(target_size=(256, 256))
    #     processor = DatasetProcessor(face_preprocessor)
        
    #     # Process dataset
    #     processor.process_dataset(
    #         raw_data_dir=RAW_DATA_DIR,
    #         processed_data_dir=PROCESSED_DATA_DIR,
    #         train_ratio=0.8
    #     )
    #     print("✅ Preprocessing completed!")
    # else:
    #     print(f"⚠️  Raw data directory '{RAW_DATA_DIR}' not found")
    #     print("   Creating dummy processed data directory...")
    #     Path(PROCESSED_DATA_DIR).mkdir(exist_ok=True)
    
    # STEP 2: INITIALIZE CLASSIFIER
    print("\n2️⃣ INITIALIZING CLASSIFIER...")
    
    try:
        pipeline = TwoStageMesoViTPipeline(
            meso_weights_path=WEIGHTS_PATH,
            vit_model_name="google/vit-base-patch16-224",
            num_classes=2,
            vit_target_size=224
        )
        print("✅ Classifier initialized!")
    except Exception as e:
        print(f"❌ Failed to initialize classifier: {e}")
        return
    
    # STEP 3: TRAINING (if processed data exists)
    print("\n3️⃣ TRAINING MODEL...")
    
    if Path(f"{PROCESSED_DATA_DIR}/train").exists():
        try:
            # Create data generator
            datagen = ImageDataGenerator(rescale=1./255)
            train_generator = datagen.flow_from_directory(
                f'{PROCESSED_DATA_DIR}/train',
                target_size=(256, 256),
                batch_size=32,
                class_mode='binary'
            )
            
            print(f"Found {train_generator.samples} training images")
            
            # Train for a few epochs
            for epoch in range(10):  # Train for 10 epochs
                print(f"Epoch {epoch + 1}/10")
                
                try:
                    X_batch, y_batch = next(train_generator)
                    
                    # Train ViT stage
                    pipeline.train_vit_stage(
                        X_batch, y_batch,
                        epochs=1,
                        batch_size=32,
                        feature_conversion='patches',
                        freeze_vit_backbone=True,
                        verbose=1
                    )
                    
                except StopIteration:
                    print("   Ran out of training data, moving to next epoch")
                    break
            
            print("✅ Training completed!")
            
        except Exception as e:
            print(f"⚠️  Training error: {e}")
    else:
        print(f"⚠️  No training data found in {PROCESSED_DATA_DIR}/train")
    
    # STEP 4: TEST PREDICTION
    print("\n4️⃣ TESTING PREDICTION...")
    
    # Test on processed data if available
    if Path(f"{PROCESSED_DATA_DIR}/test").exists():
        try:
            test_datagen = ImageDataGenerator(rescale=1./255)
            test_generator = test_datagen.flow_from_directory(
                f'{PROCESSED_DATA_DIR}/test',
                target_size=(256, 256),
                batch_size=32,
                class_mode='binary'
            )
            
            X_test, y_test = next(test_generator)
            predictions = pipeline.predict(X_test, feature_conversion='patches')
            
            print("Test Results:")
            print(f"Predictions: {predictions.flatten()}")
            print(f"True labels: {y_test.flatten()}")
            
            # Calculate accuracy
            pred_binary = (predictions > 0.5).astype(int).flatten()
            accuracy = np.mean(pred_binary == y_test.flatten())
            print(f"✅ Accuracy: {accuracy:.3f}")
            
        except Exception as e:
            print(f"⚠️  Test error: {e}")
    
    # Test on single image if provided
    if Path(TEST_IMAGE).exists():
        print(f"\n🔍 Testing single image: {TEST_IMAGE}")
        
        try:
            # Initialize face preprocessor for single image
            face_preprocessor = FacePreprocessor(target_size=(256, 256))
            
            # Load and process image
            image = cv2.imread(TEST_IMAGE)
            face = face_preprocessor.detect_and_crop_face(image)
            
            if face is not None:
                # Prepare for prediction
                face_normalized = face.astype(np.float32) / 255.0
                face_batch = np.expand_dims(face_normalized, axis=0)
                
                # Predict
                prediction = pipeline.predict(face_batch, feature_conversion='patches')
                probability = float(prediction[0][0])
                
                if probability > 0.5:
                    result = f"FAKE (confidence: {probability:.3f})"
                else:
                    result = f"REAL (confidence: {1-probability:.3f})"
                
                print(f"🎯 Result: {result}")
            else:
                print("❌ No face detected in image")
                
        except Exception as e:
            print(f"❌ Single image test error: {e}")
    
    print("\n🎉 PIPELINE COMPLETED!")

if __name__ == "__main__":
    run_complete_pipeline()