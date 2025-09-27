import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import hashlib
from scipy import ndimage
from pathlib import Path
import json
from typing import List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import time
import logging
from collections import Counter
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ImageFingerprint:
    """Image fingerprint with multiple comparison methods"""
    path: str
    # Original pyramid approach (improved)
    pyramid_hashes: List[str]
    pyramid_stats: List[dict]
    
    # New approaches for better matching
    gradient_features: dict  # Edge/gradient information
    local_features: dict    # Local descriptor statistics
    color_histogram: Optional[np.ndarray] = None  # For color images
    aspect_ratio: float = 1.0
    
    # Structural features
    dominant_orientations: List[float] = None
    texture_energy: List[float] = None

@dataclass
class ProcessingResult:
    """Wrapper for processing results, including errors"""
    success: bool
    fingerprint: Optional[ImageFingerprint] = None
    error: Optional[str] = None
    path: Optional[str] = None

class ImageProcessor:
    def __init__(self, pyramid_levels: int = 4, base_sigma: float = 1.0, workers: int = 4):
        self.pyramid_levels = pyramid_levels
        self.base_sigma = base_sigma
        self.workers = workers
        self.thread_pool = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="ImageProcessor"
        )

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.thread_pool.shutdown(wait=True)

    def process_images_threaded(self, image_paths: List[Path]) -> List[ProcessingResult]:
        """Process multiple images with  features"""
        start_time = time.time()
        logger.info(f"Starting  processing of {len(image_paths)} images")
        
        future_to_path = {}
        for path in image_paths:
            future = self.thread_pool.submit(self._process_single_image_safe, str(path.absolute()))
            future_to_path[future] = path
        
        results = []
        completed_count = 0
        
        for future in as_completed(future_to_path):
            result = future.result()
            results.append(result)
            
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == len(image_paths):
                elapsed = time.time() - start_time
                rate = completed_count / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {completed_count}/{len(image_paths)} images ({rate:.1f} img/sec)")
        
        total_time = time.time() - start_time
        success_count = sum(1 for r in results if r.success)
        logger.info(f" processing complete: {success_count}/{len(image_paths)} successful in {total_time:.1f}s")
        
        return results

    def _process_single_image_safe(self, image_path: str) -> ProcessingResult:
        """Safely process a single image with comprehensive error handling"""
        try:
            fingerprint = self._process_single_image_(image_path)
            return ProcessingResult(
                success=True,
                fingerprint=fingerprint,
                path=image_path
            )
        except Exception as e:
            logger.warning(f"Failed to process {Path(image_path).name}: {str(e)}")
            return ProcessingResult(
                success=False,
                error=str(e),
                path=image_path
            )

    def _process_single_image_(self, image_path: str) -> ImageFingerprint:
        """
         image processing with multiple feature extraction methods
        
        This comprehensive approach extracts features that are robust to:
        - Different crops and aspect ratios
        - Color variations
        - Minor rotations and scaling
        - Noise and compression artifacts
        """
        # Load image and preserve color information
        with Image.open(image_path) as img:
            # Store original color image for color features
            color_image = np.array(img.convert('RGB'))
            
            # Convert to grayscale for most processing
            if img.mode != 'L':
                gray_img = img.convert('L')
            else:
                gray_img = img
            
            image_array = np.array(gray_img)
            aspect_ratio = img.width / img.height

        # 1.  Gaussian pyramid with better hashing
        pyramid = self.create_gaussian_pyramid(image_array)
        pyramid_hashes = []
        pyramid_stats = []
        
        for level_array in pyramid:
            # Use improved hash that's more robust to cropping
            hash_val = self.create_robust_hash(level_array)
            pyramid_hashes.append(hash_val)
            
            #  statistical features
            stats = self.extract__statistical_features(level_array)
            pyramid_stats.append(stats)

        # 2. Extract gradient/edge features (robust to cropping)
        gradient_features = self.extract_gradient_features(image_array)
        
        # 3. Extract local texture features
        local_features = self.extract_local_texture_features(image_array)
        
        # 4. Color histogram (if applicable)
        color_histogram = self.extract_color_histogram(color_image)
        
        # 5. Structural features
        dominant_orientations = self.extract_dominant_orientations(image_array)
        texture_energy = self.extract_texture_energy(pyramid)

        return ImageFingerprint(
            path=image_path,
            pyramid_hashes=pyramid_hashes,
            pyramid_stats=pyramid_stats,
            gradient_features=gradient_features,
            local_features=local_features,
            color_histogram=color_histogram,
            aspect_ratio=aspect_ratio,
            dominant_orientations=dominant_orientations,
            texture_energy=texture_energy
        )

    def create_gaussian_pyramid(self, image_array: np.ndarray) -> List[np.ndarray]:
        """ Gaussian pyramid with better level selection"""
        pyramid = []
        current_image = image_array.astype(np.float64)
        
        for level in range(self.pyramid_levels):
            # More gradual sigma progression for better feature preservation
            sigma = self.base_sigma * (1.5 ** level)
            blurred = ndimage.gaussian_filter(current_image, sigma=sigma)
            pyramid.append(blurred)
            
        return pyramid

    def create_robust_hash(self, image_array: np.ndarray, hash_size: int = 16) -> str:
        """
        Create a more robust perceptual hash using DCT-based approach
        
        This method is more resilient to cropping and minor variations
        compared to simple grid-based hashing.
        """
        # Resize to standard size for consistent comparison
        # Using cv2 for better resize quality
        resized = cv2.resize(image_array, (hash_size * 4, hash_size * 4), 
                           interpolation=cv2.INTER_LANCZOS4)
        
        # Apply DCT (Discrete Cosine Transform)
        # DCT concentrates image energy in low frequencies, making it robust
        dct = cv2.dct(resized.astype(np.float32))
        
        # Keep only the top-left corner (low frequencies)
        dct_reduced = dct[:hash_size, :hash_size]
        
        # Create binary hash based on median threshold
        median_val = np.median(dct_reduced)
        binary_hash = (dct_reduced > median_val).astype(int)
        
        # Convert to string and hash
        binary_string = ''.join(binary_hash.flatten().astype(str))
        return hashlib.sha256(binary_string.encode()).hexdigest()[:16]

    def extract__statistical_features(self, image_array: np.ndarray) -> dict:
        """Extract more comprehensive statistical features"""
        flat = image_array.flatten()
        
        # Basic statistics
        features = {
            'mean': float(np.mean(flat)),
            'std': float(np.std(flat)),
            'skewness': float(self._skewness(flat)),
            'kurtosis': float(self._kurtosis(flat)),
            'energy': float(np.sum(flat ** 2))
        }
        
        # Percentile features (more robust than min/max)
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            features[f'percentile_{p}'] = float(np.percentile(flat, p))
        
        # Entropy (measure of randomness/texture)
        features['entropy'] = float(self._calculate_entropy(image_array))
        
        return features

    def extract_gradient_features(self, image_array: np.ndarray) -> dict:
        """
        Extract gradient-based features that are robust to cropping
        
        Gradients capture edge information which is less sensitive to
        exact positioning and more about structure.
        """
        # Calculate gradients using Sobel operators
        grad_x = cv2.Sobel(image_array, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image_array, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Statistical features of gradients
        features = {
            'gradient_mean': float(np.mean(magnitude)),
            'gradient_std': float(np.std(magnitude)),
            'gradient_energy': float(np.sum(magnitude**2)),
            'edge_density': float(np.sum(magnitude > np.percentile(magnitude, 75)) / magnitude.size)
        }
        
        # Gradient direction histogram (captures dominant orientations)
        hist, _ = np.histogram(direction.flatten(), bins=36, range=(-np.pi, np.pi))
        features['direction_histogram'] = hist.tolist()
        
        return features

    def extract_local_texture_features(self, image_array: np.ndarray) -> dict:
        """
        Extract Local Binary Pattern (LBP) style features
        
        LBP features are excellent for texture description and are
        relatively invariant to lighting changes and cropping.
        """
        # Simple LBP implementation
        def local_binary_pattern(img, radius=1):
            """Basic LBP implementation"""
            rows, cols = img.shape
            lbp = np.zeros_like(img)
            
            for i in range(radius, rows - radius):
                for j in range(radius, cols - radius):
                    center = img[i, j]
                    code = 0
                    
                    # Check 8 neighbors
                    neighbors = [
                        img[i-1, j-1], img[i-1, j], img[i-1, j+1],
                        img[i, j+1], img[i+1, j+1], img[i+1, j],
                        img[i+1, j-1], img[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor > center:
                            code |= (1 << k)
                    
                    lbp[i, j] = code
            
            return lbp
        
        # Extract LBP
        lbp = local_binary_pattern(image_array.astype(np.uint8))
        
        # Create histogram of LBP values
        hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0, 256))
        
        # Normalize histogram
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        
        return {
            'lbp_histogram': hist.tolist(),
            'lbp_uniformity': float(np.sum(hist**2)),  # Measure of texture uniformity
            'lbp_entropy': float(-np.sum(hist * np.log(hist + 1e-7)))  # Texture randomness
        }

    def extract_color_histogram(self, color_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract color histogram features"""
        if color_image is None or len(color_image.shape) != 3:
            return None
        
        # Calculate histogram for each channel
        histograms = []
        for channel in range(3):  # R, G, B
            hist, _ = np.histogram(color_image[:, :, channel], bins=64, range=(0, 256))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)  # Normalize
            histograms.extend(hist)
        
        return np.array(histograms)

    def extract_dominant_orientations(self, image_array: np.ndarray) -> List[float]:
        """Extract dominant edge orientations"""
        # Calculate gradients
        grad_x = cv2.Sobel(image_array, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image_array, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate orientation
        orientation = np.arctan2(grad_y, grad_x)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Weight orientations by gradient magnitude
        weighted_orientations = orientation[magnitude > np.percentile(magnitude, 80)]
        
        # Find peaks in orientation histogram
        hist, bin_edges = np.histogram(weighted_orientations, bins=36, range=(-np.pi, np.pi))
        
        # Find dominant orientations (local maxima)
        dominant = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.mean(hist):
                angle = (bin_edges[i] + bin_edges[i+1]) / 2
                dominant.append(float(angle))
        
        return dominant[:3]  # Return top 3 dominant orientations

    def extract_texture_energy(self, pyramid: List[np.ndarray]) -> List[float]:
        """Extract texture energy at different scales"""
        energies = []
        
        for level in pyramid:
            # Calculate local variance (texture measure)
            # Higher variance indicates more texture
            kernel = np.ones((5, 5)) / 25  # 5x5 averaging kernel
            local_mean = cv2.filter2D(level, -1, kernel)
            local_variance = cv2.filter2D(level**2, -1, kernel) - local_mean**2
            
            # Average texture energy
            energy = float(np.mean(local_variance))
            energies.append(energy)
        
        return energies

    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness (asymmetry measure)"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis (tail heaviness measure)"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3

    def _calculate_entropy(self, image_array: np.ndarray) -> float:
        """Calculate image entropy (measure of information content)"""
        # Calculate histogram
        hist, _ = np.histogram(image_array.flatten(), bins=256, range=(0, 256))
        
        # Normalize to get probabilities
        hist = hist.astype(float)
        hist /= hist.sum()
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log(hist + 1e-7))
        return entropy


class ImageMatcher:
    """
     image matcher with multiple comparison strategies
    """
    
    def __init__(self, weights: dict[str, float] = None):
        # Default weights for different feature types
        self.weights = weights or {
            'pyramid_hash': 0.3,
            'pyramid_stats': 0.2,
            'gradient': 0.2,
            'texture': 0.15,
            'color': 0.1,
            'structural': 0.05
        }
    
    def compare_fingerprints(self, fp1: ImageFingerprint, 
                           fp2: ImageFingerprint) -> float:
        """
        Comprehensive fingerprint comparison using multiple feature types
        
        Returns distance score (0.0 = identical, 1.0 = completely different)
        """
        distances = {}
        
        # 1. Pyramid hash comparison (improved)
        distances['pyramid_hash'] = self._compare_pyramid_hashes(fp1, fp2)
        
        # 2. Statistical features comparison
        distances['pyramid_stats'] = self._compare_pyramid_stats(fp1, fp2)
        
        # 3. Gradient features comparison
        distances['gradient'] = self._compare_gradient_features(fp1, fp2)
        
        # 4. Texture features comparison
        distances['texture'] = self._compare_texture_features(fp1, fp2)
        
        # 5. Color comparison (if available)
        distances['color'] = self._compare_color_features(fp1, fp2)
        
        # 6. Structural features comparison
        distances['structural'] = self._compare_structural_features(fp1, fp2)
        
        # Weighted combination
        total_distance = 0
        total_weight = 0
        
        for feature_type, distance in distances.items():
            if distance is not None:  # Some features might not be available
                weight = self.weights.get(feature_type, 0)
                total_distance += weight * distance
                total_weight += weight
        
        if total_weight == 0:
            return 1.0  # No valid comparisons
        
        return total_distance / total_weight

    def _compare_pyramid_hashes(self, fp1: ImageFingerprint, 
                               fp2: ImageFingerprint) -> float:
        """Compare DCT-based pyramid hashes"""
        hash_distances = []
        
        for i in range(min(len(fp1.pyramid_hashes), len(fp2.pyramid_hashes))):
            distance = self.hamming_distance(fp1.pyramid_hashes[i], fp2.pyramid_hashes[i])
            hash_distances.append(distance)
        
        return np.mean(hash_distances) if hash_distances else 1.0

    def _compare_pyramid_stats(self, fp1: ImageFingerprint, 
                              fp2: ImageFingerprint) -> float:
        """Compare statistical features across pyramid levels"""
        stat_distances = []
        
        for i in range(min(len(fp1.pyramid_stats), len(fp2.pyramid_stats))):
            distance = self.statistical_distance(fp1.pyramid_stats[i], fp2.pyramid_stats[i])
            stat_distances.append(distance)
        
        return np.mean(stat_distances) if stat_distances else 1.0

    def _compare_gradient_features(self, fp1: ImageFingerprint, 
                                  fp2: ImageFingerprint) -> float:
        """Compare gradient-based features"""
        # Compare scalar gradient features
        scalar_distance = self.statistical_distance(
            {k: v for k, v in fp1.gradient_features.items() if k != 'direction_histogram'},
            {k: v for k, v in fp2.gradient_features.items() if k != 'direction_histogram'}
        )
        
        # Compare direction histograms using histogram intersection
        hist1 = np.array(fp1.gradient_features.get('direction_histogram', []))
        hist2 = np.array(fp2.gradient_features.get('direction_histogram', []))
        
        if len(hist1) > 0 and len(hist2) > 0:
            # Normalize histograms
            hist1 = hist1 / (np.sum(hist1) + 1e-7)
            hist2 = hist2 / (np.sum(hist2) + 1e-7)
            
            # Use histogram intersection (higher = more similar)
            intersection = np.sum(np.minimum(hist1, hist2))
            hist_distance = 1.0 - intersection
        else:
            hist_distance = 1.0
        
        # Combine scalar and histogram distances
        return (scalar_distance + hist_distance) / 2

    def _compare_texture_features(self, fp1: ImageFingerprint, 
                                 fp2: ImageFingerprint) -> float:
        """Compare texture features"""
        # Compare LBP histograms
        hist1 = np.array(fp1.local_features.get('lbp_histogram', []))
        hist2 = np.array(fp2.local_features.get('lbp_histogram', []))
        
        if len(hist1) > 0 and len(hist2) > 0:
            # Chi-squared distance for histograms
            chi_squared = np.sum((hist1 - hist2)**2 / (hist1 + hist2 + 1e-7))
            hist_distance = chi_squared / 2  # Normalize
        else:
            hist_distance = 1.0
        
        # Compare other texture features
        scalar_features = ['lbp_uniformity', 'lbp_entropy']
        scalar_dict1 = {k: fp1.local_features[k] for k in scalar_features if k in fp1.local_features}
        scalar_dict2 = {k: fp2.local_features[k] for k in scalar_features if k in fp2.local_features}
        
        scalar_distance = self.statistical_distance(scalar_dict1, scalar_dict2)
        
        return (hist_distance + scalar_distance) / 2

    def _compare_color_features(self, fp1: ImageFingerprint, 
                               fp2: ImageFingerprint) -> Optional[float]:
        """Compare color histogram features"""
        if fp1.color_histogram is None or fp2.color_histogram is None:
            return None
        
        # Normalize histograms
        hist1 = fp1.color_histogram / (np.sum(fp1.color_histogram) + 1e-7)
        hist2 = fp2.color_histogram / (np.sum(fp2.color_histogram) + 1e-7)
        
        # Use histogram intersection
        intersection = np.sum(np.minimum(hist1, hist2))
        return 1.0 - intersection

    def _compare_structural_features(self, fp1: ImageFingerprint, 
                                   fp2: ImageFingerprint) -> float:
        """Compare structural features like orientations and texture energy"""
        distances = []
        
        # Compare dominant orientations
        if fp1.dominant_orientations and fp2.dominant_orientations:
            # Simple approach: compare the most dominant orientation
            angle_diff = abs(fp1.dominant_orientations[0] - fp2.dominant_orientations[0])
            # Handle circular nature of angles
            angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
            orientation_distance = angle_diff / np.pi  # Normalize to [0, 1]
            distances.append(orientation_distance)
        
        # Compare texture energy across scales
        if fp1.texture_energy and fp2.texture_energy:
            energy_distances = []
            for i in range(min(len(fp1.texture_energy), len(fp2.texture_energy))):
                e1, e2 = fp1.texture_energy[i], fp2.texture_energy[i]
                max_energy = max(abs(e1), abs(e2), 1e-10)
                energy_distances.append(abs(e1 - e2) / max_energy)
            
            if energy_distances:
                distances.append(np.mean(energy_distances))
        
        # Compare aspect ratios
        aspect_diff = abs(fp1.aspect_ratio - fp2.aspect_ratio)
        aspect_distance = min(aspect_diff / max(fp1.aspect_ratio, fp2.aspect_ratio), 1.0)
        distances.append(aspect_distance)
        
        return np.mean(distances) if distances else 1.0

    def hamming_distance(self, hash1: str, hash2: str) -> float:
        """Calculate normalized Hamming distance between two hashes"""
        if len(hash1) != len(hash2):
            return 1.0
        
        # Convert hex hashes to binary for bit-wise comparison
        bin1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
        bin2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)
        
        differences = sum(b1 != b2 for b1, b2 in zip(bin1, bin2))
        return differences / len(bin1)
    
    def statistical_distance(self, stats1: dict, stats2: dict) -> float:
        """Calculate normalized distance between statistical features"""
        distances = []
        
        for key in stats1.keys():
            if key in stats2:
                val1, val2 = stats1[key], stats2[key]
                max_val = max(abs(val1), abs(val2), 1e-10)
                normalized_diff = abs(val1 - val2) / max_val
                distances.append(normalized_diff)
        
        return np.mean(distances) if distances else 1.0

    def find_matches(self, set1_fingerprints: List[ImageFingerprint], 
                    set2_fingerprints: List[ImageFingerprint], 
                    threshold: float = 0.4) -> List[Tuple[str, str, float]]:
        """
        Find matching images with improved algorithm
        
        Args:
            set1_fingerprints: First set of fingerprints
            set2_fingerprints: Second set of fingerprints
            threshold: Maximum distance for considering a match
            
        Returns:
            List of (path1, path2, distance) tuples for matches
        """
        matches = []
        
        for fp1 in set1_fingerprints:
            best_match = None
            best_distance = float('inf')
            
            for fp2 in set2_fingerprints:
                distance = self.compare_fingerprints(fp1, fp2)
                
                if distance < best_distance and distance <= threshold:
                    best_distance = distance
                    best_match = fp2
            
            if best_match:
                matches.append((fp1.path, best_match.path, best_distance))
        
        # Sort by similarity (lower distance = higher similarity)
        matches.sort(key=lambda x: x[2])
        
        return matches


# Utility functions
def extract_successful_fingerprints(results: List[ProcessingResult]) -> List[ImageFingerprint]:
    """Extract successful fingerprints from processing results"""
    return [result.fingerprint for result in results if result.success and result.fingerprint]

def report_processing_errors(results: List[ProcessingResult]) -> None:
    """Report processing errors"""
    errors = [result for result in results if not result.success]
    if errors:
        logger.warning(f"Processing failed for {len(errors)} images:")
        for error_result in errors[:5]:
            logger.warning(f"  {Path(error_result.path).name}: {error_result.error}")
        if len(errors) > 5:
            logger.warning(f"  ... and {len(errors) - 5} more errors")

def get_mapping_file(filepath: Path) -> dict[Any]:
    data = {}

    with open(filepath.absolute(), "r+") as file:
        lines = file.readlines()
        counter = 0
        for line in lines:
            if counter == 0:
                counter += 1
                continue
            line_data = line.split("\t")
            data[line_data[0]] = [line_data[1], line_data[2], line_data[3], line_data[4], line_data[5], line_data[6], line_data[7], line_data[8]]
            counter += 1

    return data


def main():
    """ main function with better matching"""
    # Define your image paths
    kb_paths = Path(r"C:\SHARES\Development\HACK4DK\DAMGAARD\Damgaard\jpegDamgaardResized")
    kb_paths = [path for path in kb_paths.rglob("*.jpg")]
    pol_paths = Path(r"C:\SHARES\Development\HACK4DK\holger\previews\combi")
    pol_paths = [path for path in pol_paths.rglob("*.jpg")]
    
    # Use  processor with context manager
    with ImageProcessor(pyramid_levels=4, workers=4) as processor:
        # Create matcher with custom weights if needed
        matcher = ImageMatcher()
        
        # Process first set
        print("Processing KB image set...")
        kb_results = processor.process_images_threaded(kb_paths)
        kb_fingerprints = extract_successful_fingerprints(kb_results)
        report_processing_errors(kb_results)
        
        # Process second set
        print("Processing POL image set...")
        pol_results = processor.process_images_threaded(pol_paths)
        pol_fingerprints = extract_successful_fingerprints(pol_results)
        report_processing_errors(pol_results)
        
        # Find matches with lower threshold for better sensitivity
        print("\nFinding matches...")
        matches = matcher.find_matches(kb_fingerprints, pol_fingerprints, threshold=1.0)
        
        print(f"\nFound {len(matches)} matches:")

        pol_metadata = get_mapping_file(Path(r"C:\SHARES\Development\HACK4DK\holger\Holger Damgaard Metadata.txt"))
        data_path = r"C:\SHARES\HACK4DK\matches.csv"
        with open(data_path, "w+") as file:
            file.write("KB_File;Pol_File;Similarity;ScanpixID;SupplierID;Caption;Headline;Keywords;City;MY14;CreateDate;Width x Height\n")
            for path1, path2, distance in matches:
                similarity_percent = (1 - distance) * 100
                key = Path(path2).name.split(".jpg")[0]
                metadata = pol_metadata.get(key, 'Not found')
                print(f"{Path(path1).name} ↔ {Path(path2).name} (similarity: {similarity_percent:.2f}%)")

                if metadata == "Not found":
                    file.write(f"{path1};{path2};{similarity_percent:.1f}%;Kun ikke finde pol filen til metadata;;;;;;;;\n")
                else:
                    file.write(f"{path1};{path2};{similarity_percent:.1f}%;{metadata[0]};{metadata[1]};{metadata[2]};{metadata[3]};{metadata[4]};{metadata[5]};{metadata[6]};{metadata[7]};{metadata[8]}\n")

if __name__ == "__main__":
    main()
