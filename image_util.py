
import os
import sys
from pathlib import Path
from io import BytesIO
import numpy as np
from PIL import Image
from rembg import remove, new_session

def generate_image_mask(input_path):
    """Generate mask from input image (0=background, 255=object)"""
    try:
        if not os.path.exists(input_path):
            print(f"❌ Image does not exist: {input_path}")
            return None
        
        
        session = new_session('u2net')
        
        
        with open(input_path, 'rb') as f:
            input_data = f.read()
        
       
        output_data = remove(input_data, session=session)
        
       
        output_image = Image.open(BytesIO(output_data))
        
        
        alpha_channel = output_image.split()[-1]
        mask_array = np.array(alpha_channel)
        
        
        binary_mask = np.where(mask_array == 0, 0, 1).astype(np.uint8)
        
        
        output_dir = Path("test_image_mask")
        output_dir.mkdir(exist_ok=True)
        
        
        input_path_obj = Path(input_path)
        transparent_filename = f"transparent_{input_path_obj.stem}.png"
        transparent_path = output_dir / transparent_filename
        output_image.save(transparent_path, format='PNG')
        
        
        mask_filename = f"mask_{input_path_obj.stem}.png"
        mask_path = output_dir / mask_filename
        
        
        mask_for_display = np.where(binary_mask == 1, 0, 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_for_display, mode='L')
        mask_image.save(mask_path)
        
        return binary_mask
        
    except Exception as e:
        print(f"❌ Failed to process: {e}")
        return None

