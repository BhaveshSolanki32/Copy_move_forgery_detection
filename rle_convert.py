import os
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import sys

# ==============================================================================
# 1. DECODING LOGIC (From your "Incorrect" Script)
# ==============================================================================
def rle_decode(mask_rle_input, shape):
    """
    Decodes the RLE string (JSON format [start, len, ...]) into a binary mask.
    Args:
        mask_rle_input: String or List containing RLE data.
        shape: (height, width) of the target image.
    """
    try:
        # Handle format variations (String vs List)
        if isinstance(mask_rle_input, str):
            if mask_rle_input == '-' or not mask_rle_input:
                return np.zeros(shape, dtype=np.uint8)
            mask_rle = json.loads(mask_rle_input)
        else:
            mask_rle = mask_rle_input

        # Create flat array
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        
        # Extract starts and lengths
        # The format is [start, length, start, length, ...]
        starts = np.array(mask_rle[0::2])
        lengths = np.array(mask_rle[1::2])
        
        # 1-based indexing to 0-based
        starts -= 1
        ends = starts + lengths
        
        # Set pixels
        for lo, hi in zip(starts, ends):
            # Boundary check to be safe
            if lo < 0: lo = 0
            if hi > len(img): hi = len(img)
            img[lo:hi] = 1
            
        # Reshape (Fortran order 'F' is standard for these RLEs)
        return img.reshape(shape, order='F')
        
    except Exception as e:
        print(f"Warning: RLE Decode error: {e}")
        return np.zeros(shape, dtype=np.uint8)

# ==============================================================================
# 2. CONVERSION WORKER
# ==============================================================================
def convert_rle_to_raw_masks(src_mask_dir, img_dir, dest_mask_dir):
    """
    Reads RLE-string .npy files, converts them to raw numpy arrays (N, H, W),
    and saves them to dest_mask_dir.
    """
    if not os.path.exists(src_mask_dir):
        print(f"❌ Source mask directory not found: {src_mask_dir}")
        return
    if not os.path.exists(img_dir):
        print(f"❌ Image directory not found (needed for dimensions): {img_dir}")
        return
        
    # Create destination folder
    os.makedirs(dest_mask_dir, exist_ok=True)
    
    # List files
    mask_files = [f for f in os.listdir(src_mask_dir) if f.endswith('.npy')]
    print(f"🔄 Found {len(mask_files)} masks to convert...")
    
    success_count = 0
    error_count = 0
    
    for mask_file in tqdm(mask_files, desc="Converting"):
        base_name = os.path.splitext(mask_file)[0]
        
        # Paths
        src_path = os.path.join(src_mask_dir, mask_file)
        dest_path = os.path.join(dest_mask_dir, mask_file)
        img_path = os.path.join(img_dir, f"{base_name}.png")
        
        # 1. Get Dimensions from Image
        if not os.path.exists(img_path):
            # Try .jpg if .png missing
            img_path = os.path.join(img_dir, f"{base_name}.jpg")
            if not os.path.exists(img_path):
                # print(f"⚠️ Image missing for {mask_file}, skipping...")
                error_count += 1
                continue
        
        try:
            with Image.open(img_path) as pil_img:
                W, H = pil_img.size
        except Exception as e:
            print(f"⚠️ Error reading image {base_name}: {e}")
            error_count += 1
            continue
            
        # 2. Load Source RLE
        try:
            rle_payload = np.load(src_path, allow_pickle=True)
            
            # Extract string content
            if rle_payload.shape == ():
                rle_content = rle_payload.item()
            else:
                rle_content = str(rle_payload)
                
            # Handle list of strings (if saved that way) or semicolon separated string
            rle_strings = []
            if isinstance(rle_content, str):
                rle_strings = rle_content.split(';')
            elif isinstance(rle_content, (list, np.ndarray)):
                 # If it's a list like ["RLE1", "RLE2"]
                rle_strings = rle_content
            
        except Exception as e:
            print(f"⚠️ Error loading/parsing npy {base_name}: {e}")
            error_count += 1
            continue
            
        # 3. Decode into (N, H, W) Array
        decoded_masks = []
        for rle_str in rle_strings:
            if not rle_str or rle_str == '-': 
                continue
            
            # Decode
            mask = rle_decode(rle_str, (H, W))
            
            # Only add if mask has pixels
            if np.sum(mask) > 0:
                decoded_masks.append(mask)
        
        if not decoded_masks:
            # If no valid masks, save an empty zeros array (H, W) or skip?
            # Saving empty array (1, H, W) of zeros is safer
            final_array = np.zeros((1, H, W), dtype=np.uint8)
        else:
            # Stack into (N, H, W)
            final_array = np.stack(decoded_masks, axis=0).astype(np.uint8)
            
        # 4. Save to Destination
        np.save(dest_path, final_array)
        success_count += 1

    print("\n✅ Conversion Complete.")
    print(f"   Success: {success_count}")
    print(f"   Errors/Skipped: {error_count}")
    print(f"   Saved to: {dest_mask_dir}")

# ==============================================================================
# 3. RUN IT
# ==============================================================================

# --- CONFIGURE PATHS HERE ---
WRONG_MASK_DIR = r"synthetic_forgery_dataset_05\masks"
IMG_DIR = r"synthetic_forgery_dataset_05\images"

# New folder where corrected masks will be saved
NEW_MASK_DIR = r"synthetic_forgery_dataset_05\masks_converted"

convert_rle_to_raw_masks(WRONG_MASK_DIR, IMG_DIR, NEW_MASK_DIR)