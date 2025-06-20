import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.optimizers import Adam
from transformers import TFAutoModel, AutoConfig, AutoImageProcessor
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMGWIDTH = 256

class Classifier:
    def __init__(self):
        self.model = 0
    
    def predict(self, x):
        if x.size == 0:
            return []
        return self.model.predict(x)
    
    def fit(self, x, y):
        return self.model.train_on_batch(x, y)
    
    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)
    
    def load(self, path):
        self.model.load_weights(path)


class Meso4(Classifier):
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        self.feature_extractor = self.init_model(return_features=True)
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    
    def init_model(self, return_features=False): 
        x = Input(shape=(IMGWIDTH, IMGWIDTH, 3))
        
        x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        
        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        if return_features:
            return KerasModel(inputs=x, outputs=x4)
        
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(negative_slope=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        return KerasModel(inputs=x, outputs=y)
    
    def get_meso_features(self, x):
        """Extract intermediate features from Meso4"""
        return self.feature_extractor.predict(x)


class FeatureToImageConverter:
    """
    Converts Meso4 features to ViT-compatible images
    """
    def __init__(self, target_size=224):
        self.target_size = target_size
    
    def features_to_patches(self, features):
        """
        Convert Meso4 features [batch_size, 8, 8, 16] to patch-like images
        """
        batch_size = features.shape[0]
        
        # Method 1: Spatial upsampling with channel reduction
        # Upsample from 8x8 to target size and reduce channels from 16 to 3
        upsampled = tf.image.resize(features, [self.target_size, self.target_size])
        
        # Reduce channels from 16 to 3 using a learned projection or simple averaging
        if features.shape[-1] == 16:
            # Take every 5th channel and average some to get RGB-like representation
            channel_1 = tf.reduce_mean(upsampled[:, :, :, 0:6], axis=-1, keepdims=True)
            channel_2 = tf.reduce_mean(upsampled[:, :, :, 6:11], axis=-1, keepdims=True)
            channel_3 = tf.reduce_mean(upsampled[:, :, :, 11:16], axis=-1, keepdims=True)
            rgb_features = tf.concat([channel_1, channel_2, channel_3], axis=-1)
        else:
            rgb_features = upsampled
        
        # Normalize to [0, 1] range
        rgb_features = tf.nn.sigmoid(rgb_features)
        
        return rgb_features
    
    def features_to_grid(self, features):
        """
        Alternative: Create a grid visualization of feature maps
        """
        batch_size, h, w, channels = features.shape
        
        # Create a grid of feature maps
        grid_size = int(np.ceil(np.sqrt(channels)))
        
        # Pad channels to make a perfect square
        pad_channels = grid_size * grid_size - channels
        if pad_channels > 0:
            padding = tf.zeros([batch_size, h, w, pad_channels])
            features_padded = tf.concat([features, padding], axis=-1)
        else:
            features_padded = features
        
        # Reshape to create grid
        features_grid = tf.reshape(features_padded, [batch_size, h, w, grid_size, grid_size])
        features_grid = tf.transpose(features_grid, [0, 1, 3, 2, 4])
        features_grid = tf.reshape(features_grid, [batch_size, h * grid_size, w * grid_size])
        
        # Convert to 3-channel by repeating
        features_rgb = tf.stack([features_grid, features_grid, features_grid], axis=-1)
        
        # Resize to target size
        features_resized = tf.image.resize(features_rgb, [self.target_size, self.target_size])
        
        # Normalize
        features_normalized = tf.nn.sigmoid(features_resized)
        
        return features_resized
    
class AlternativeViTClassifier:
    """
    Alternative ViT classifier that avoids symbolic tensor issues
    """
    def __init__(self, model_name="google/vit-base-patch16-224", num_classes=2, learning_rate=0.001):
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pretrained ViT and its processor
        self.config = AutoConfig.from_pretrained(model_name)
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.vit_backbone = TFAutoModel.from_pretrained(model_name, config=self.config)
        
        # Build only the classification head as a separate model
        self.classification_head = self._build_classification_head()
        
        optimizer = Adam(learning_rate=learning_rate)
        self.classification_head.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )
    
    def _build_classification_head(self):
        """Build only the classification head"""
        # Input is the ViT CLS token features (768 dimensions for base ViT)
        vit_features_input = Input(shape=(768,))  # ViT-base hidden size
        
        # Classification layers
        x = Dense(512, activation='relu', name='classifier_dense1')(vit_features_input)
        x = Dropout(0.3, name='classifier_dropout1')(x)
        x = Dense(256, activation='relu', name='classifier_dense2')(x)
        x = Dropout(0.3, name='classifier_dropout2')(x)
        
        if self.num_classes == 2:
            output = Dense(1, activation='sigmoid', name='classifier_output')(x)
        else:
            output = Dense(self.num_classes, activation='softmax', name='classifier_output')(x)
        
        return KerasModel(inputs=vit_features_input, outputs=output)
    
    def preprocess_for_vit(self, images):
        """Properly preprocess images for ViT"""
        # Convert to numpy if it's a tensor
        if isinstance(images, tf.Tensor):
            images_np = images.numpy()
        else:
            images_np = images
        
        # Ensure images are in [0, 255] range for the processor
        if images_np.max() <= 1.0:
            images_np = images_np * 255.0
        
        # Convert to uint8
        images_np = images_np.astype(np.uint8)
        
        # Process each image individually and collect results
        processed_images = []
        for i in range(images_np.shape[0]):
            # Use the HuggingFace processor
            processed = self.image_processor(
                images_np[i], 
                return_tensors="tf",
                do_rescale=True,
                do_normalize=True
            )
            processed_images.append(processed['pixel_values'][0])
        
        # Stack all processed images
        return tf.stack(processed_images, axis=0)
    
    def get_vit_features(self, images):
        """Extract features from ViT backbone"""
        # Preprocess images properly for ViT
        processed_images = self.preprocess_for_vit(images)
        
        # Extract features using the preprocessed images
        vit_outputs = self.vit_backbone(pixel_values=processed_images, training=False)
        return vit_outputs.last_hidden_state[:, 0]  # CLS token
    
    def predict(self, images):
        """Two-step prediction: ViT features -> classification"""
        vit_features = self.get_vit_features(images)
        return self.classification_head.predict(vit_features)
    
    def fit(self, images, labels, epochs=10, batch_size=16, validation_data=None, **kwargs):
        """Two-step training: extract features then train classifier"""
        # Extract ViT features
        print("Extracting ViT features...")
        vit_features = self.get_vit_features(images)
        
        # Handle validation data if provided
        validation_features = None
        if validation_data is not None:
            val_images, val_labels = validation_data
            validation_features = (self.get_vit_features(val_images), val_labels)
        
        # Train classification head
        print("Training classification head...")
        return self.classification_head.fit(
            vit_features, labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_features,
            **kwargs
        )
    
    def freeze_backbone(self):
        """Freeze ViT backbone"""
        self.vit_backbone.trainable = False
    
    def unfreeze_backbone(self):
        """Unfreeze ViT backbone"""
        self.vit_backbone.trainable = True


class TwoStageMesoViTPipeline:
    """
    Two-stage pipeline: Meso4 feature extraction -> ViT classification
    """
    def __init__(self, meso_weights_path=None, vit_model_name="google/vit-base-patch16-224", 
                 num_classes=2, vit_target_size=224):
        
        # Stage 1: Meso4 feature extractor
        self.meso_extractor = Meso4()
        if meso_weights_path:
            self.meso_extractor.load(meso_weights_path)
        
        # Feature converter
        self.feature_converter = FeatureToImageConverter(target_size=vit_target_size)
        
        # Stage 2: ViT classifier (using alternative implementation)
        self.vit_classifier = AlternativeViTClassifier(
            model_name=vit_model_name,
            num_classes=num_classes
        )
        
        self.num_classes = num_classes
    
    def extract_features(self, images):
        """Stage 1: Extract Meso4 features"""
        return self.meso_extractor.get_meso_features(images)
    
    def convert_features(self, meso_features, method='patches'):
        """Convert Meso4 features to ViT-compatible format"""
        if method == 'patches':
            return self.feature_converter.features_to_patches(meso_features)
        elif method == 'grid':
            return self.feature_converter.features_to_grid(meso_features)
        else:
            raise ValueError("Method must be 'patches' or 'grid'")
    
    def classify_features(self, vit_features):
        """Stage 2: Classify using ViT"""
        return self.vit_classifier.predict(vit_features)
    
    def predict(self, images, feature_conversion='patches'):
        """End-to-end prediction"""
        # Stage 1: Extract Meso4 features
        meso_features = self.extract_features(images)
        
        # Convert features to ViT format
        vit_features = self.convert_features(meso_features, method=feature_conversion)
        
        # Stage 2: ViT classification
        predictions = self.classify_features(vit_features)
        
        return predictions
    
    def train_vit_stage(self, images, labels, epochs=10, batch_size=16, 
                       feature_conversion='patches', freeze_vit_backbone=True, **kwargs):
        """Train only the ViT stage with frozen Meso4"""
        
        # Extract and convert features
        meso_features = self.extract_features(images)
        vit_features = self.convert_features(meso_features, method=feature_conversion)
        
        # Optionally freeze ViT backbone
        if freeze_vit_backbone:
            self.vit_classifier.freeze_backbone()
        
        # Train ViT classifier
        history = self.vit_classifier.fit(
            vit_features, labels, 
            epochs=epochs, 
            batch_size=batch_size,
            **kwargs
        )
        
        return history
    
    def get_intermediate_outputs(self, images):
        """Get outputs from each stage for analysis"""
        meso_features = self.extract_features(images)
        vit_patches = self.convert_features(meso_features, method='patches')
        vit_grid = self.convert_features(meso_features, method='grid')
        predictions = self.classify_features(vit_patches)
        
        return {
            'meso_features': meso_features,
            'vit_patches': vit_patches,
            'vit_grid': vit_grid,
            'predictions': predictions
        }


# # Example usage
# def main():
#     # Initialize the two-stage pipeline
#     pipeline = TwoStageMesoViTPipeline(
#         meso_weights_path='weights/Meso4_DF.h5',  # Your pretrained Meso4 weights
#         vit_model_name="google/vit-base-patch16-224",
#         num_classes=2,
#         vit_target_size=224
#     )
    
#     # Load your data
#     dataGenerator = ImageDataGenerator(rescale=1./255)
#     generator = dataGenerator.flow_from_directory(
#         'test_images',
#         target_size=(256, 256),
#         batch_size=8,
#         class_mode='binary',
#         subset='training'
#     )
    
#     # Get a batch for testing
#     X, y = next(generator)
    
#     print("Original images shape:", X.shape)
    
#     # Stage 1: Extract Meso4 features
#     meso_features = pipeline.extract_features(X)
#     print("Meso4 features shape:", meso_features.shape)
    
#     # Convert features for ViT
#     vit_patches = pipeline.convert_features(meso_features, method='patches')
#     vit_grid = pipeline.convert_features(meso_features, method='grid')
#     print("ViT patches shape:", vit_patches.shape)
#     print("ViT grid shape:", vit_grid.shape)
    
#     # End-to-end prediction
#     predictions = pipeline.predict(X, feature_conversion='patches')
#     print("Pipeline predictions shape:", predictions.shape)
#     print("Predictions:", predictions.flatten())
#     print("True labels:", y.flatten())
    
#     # Train only the ViT stage
#     print("\nTraining ViT stage...")
#     history = pipeline.train_vit_stage(
#         X, y, 
#         epochs=2, 
#         batch_size=4, 
#         feature_conversion='patches',
#         freeze_vit_backbone=True
#     )
    
#     # Get intermediate outputs for analysis
#     intermediate = pipeline.get_intermediate_outputs(X[:2])  # Just first 2 samples
#     print("\nIntermediate outputs shapes:")
#     for key, value in intermediate.items():
#         print(f"{key}: {value.shape}")


#     # Get another batch for testing
#     X, y = next(generator)
    
#     print("Original images shape:", X.shape)
    
#     # Stage 1: Extract Meso4 features
#     meso_features = pipeline.extract_features(X)
#     print("Meso4 features shape:", meso_features.shape)
    
#     # Convert features for ViT
#     vit_patches = pipeline.convert_features(meso_features, method='patches')
#     vit_grid = pipeline.convert_features(meso_features, method='grid')
#     print("ViT patches shape:", vit_patches.shape)
#     print("ViT grid shape:", vit_grid.shape)
    
#     # End-to-end prediction
#     predictions = pipeline.predict(X, feature_conversion='patches')
#     print("Pipeline predictions shape:", predictions.shape)
#     print("Predictions:", predictions.flatten())
#     print("True labels:", y.flatten())


# if __name__ == "__main__":
#     main()