<<<<<<< HEAD
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def load_mri(img_path):
    """Load 3D MRI scan using NiBabel."""
    img = nib.load(img_path)
    return img.get_fdata()

def normalize(data):
    """Normalize data to uint8 format."""
    data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)
    
    data_min = np.min(data)
    data_max = np.max(data)
    
    if data_min == data_max:
        return np.zeros_like(data, dtype=np.uint8)
    
    return ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)

def is_empty_slice(slice_data, threshold=0.02):
    """Removes the first and last few slices"""
    slice_normalized = slice_data / (np.max(slice_data) + 1e-10)
    
    if np.max(slice_normalized) < threshold:
        return True
    
    meaningful_pixels = np.sum(slice_normalized > threshold/5) / slice_data.size
    if meaningful_pixels < 0.05:  
        return True
        
    if np.std(slice_normalized) < threshold/10:
        return True
        
    return False

def extract_slices(data, output_dir):
    """Extract and save 2D slices from 3D MRI scan."""
    axes = ['axial', 'coronal', 'sagittal']
    for axis_idx, axis_name in enumerate(axes):
        axis_dir = os.path.join(output_dir, axis_name)
        os.makedirs(axis_dir, exist_ok=True)
        
        n_slices = data.shape[axis_idx]
        
        first_valid_slice = 0
        last_valid_slice = n_slices - 1
        
        for i in range(n_slices):
            if axis_idx == 0:
                slice_data = data[i, :, :]
            elif axis_idx == 1:
                slice_data = data[:, i, :]
            else:
                slice_data = data[:, :, i]
                
            if not is_empty_slice(slice_data):
                first_valid_slice = i
                break
        
        for i in range(n_slices - 1, -1, -1):
            if axis_idx == 0:
                slice_data = data[i, :, :]
            elif axis_idx == 1:
                slice_data = data[:, i, :]
            else:
                slice_data = data[:, :, i]
                
            if not is_empty_slice(slice_data):
                last_valid_slice = i
                break
        
        print(f"{axis_name}: Using slices {first_valid_slice} to {last_valid_slice} (skipping {first_valid_slice} slices at start, {n_slices - last_valid_slice - 1} at end)")
        
        slice_count = 0
        for slice_idx in range(first_valid_slice, last_valid_slice + 1):
            if axis_idx == 0:
                slice_data = data[slice_idx, :, :]
            elif axis_idx == 1:
                slice_data = data[:, slice_idx, :]
            else:
                slice_data = data[:, :, slice_idx]
            
            if is_empty_slice(slice_data):
                continue
                
            slice_normalized = normalize(slice_data)
            slice_normalized = np.squeeze(slice_normalized)
            
            plt.figure(figsize=(10, 10), frameon=False)
            plt.imshow(slice_normalized, cmap='gray')
            plt.axis('off')
            plt.tight_layout(pad=0)
            output_path = os.path.join(axis_dir, f'slice_{slice_count:04d}.png')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close()
            
            slice_count += 1
        
        print(f"Saved {slice_count} non-empty {axis_name} slices to {axis_dir}")

    print(f"All slices have been extracted and saved to: {output_dir}")

def main():
    img_path = "path/to/mri.img"
    output_dir = "sliced_mri"
    data = load_mri(img_path)
    extract_slices(data, output_dir)

if __name__ == "__main__":
    main()
=======
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def load_mri(img_path):
    """Load 3D MRI scan using NiBabel."""
    img = nib.load(img_path)
    return img.get_fdata()

def normalize(data):
    """Normalize data to uint8 format."""
    data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)
    
    data_min = np.min(data)
    data_max = np.max(data)
    
    if data_min == data_max:
        return np.zeros_like(data, dtype=np.uint8)
    
    return ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)

def is_empty_slice(slice_data, threshold=0.02):
    """Removes the first and last few slices"""
    slice_normalized = slice_data / (np.max(slice_data) + 1e-10)
    
    if np.max(slice_normalized) < threshold:
        return True
    
    meaningful_pixels = np.sum(slice_normalized > threshold/5) / slice_data.size
    if meaningful_pixels < 0.05:  
        return True
        
    if np.std(slice_normalized) < threshold/10:
        return True
        
    return False

def extract_slices(data, output_dir):
    """Extract and save 2D slices from 3D MRI scan."""
    axes = ['axial', 'coronal', 'sagittal']
    for axis_idx, axis_name in enumerate(axes):
        axis_dir = os.path.join(output_dir, axis_name)
        os.makedirs(axis_dir, exist_ok=True)
        
        n_slices = data.shape[axis_idx]
        
        first_valid_slice = 0
        last_valid_slice = n_slices - 1
        
        for i in range(n_slices):
            if axis_idx == 0:
                slice_data = data[i, :, :]
            elif axis_idx == 1:
                slice_data = data[:, i, :]
            else:
                slice_data = data[:, :, i]
                
            if not is_empty_slice(slice_data):
                first_valid_slice = i
                break
        
        for i in range(n_slices - 1, -1, -1):
            if axis_idx == 0:
                slice_data = data[i, :, :]
            elif axis_idx == 1:
                slice_data = data[:, i, :]
            else:
                slice_data = data[:, :, i]
                
            if not is_empty_slice(slice_data):
                last_valid_slice = i
                break
        
        print(f"{axis_name}: Using slices {first_valid_slice} to {last_valid_slice} (skipping {first_valid_slice} slices at start, {n_slices - last_valid_slice - 1} at end)")
        
        slice_count = 0
        for slice_idx in range(first_valid_slice, last_valid_slice + 1):
            if axis_idx == 0:
                slice_data = data[slice_idx, :, :]
            elif axis_idx == 1:
                slice_data = data[:, slice_idx, :]
            else:
                slice_data = data[:, :, slice_idx]
            
            if is_empty_slice(slice_data):
                continue
                
            slice_normalized = normalize(slice_data)
            slice_normalized = np.squeeze(slice_normalized)
            
            plt.figure(figsize=(10, 10), frameon=False)
            plt.imshow(slice_normalized, cmap='gray')
            plt.axis('off')
            plt.tight_layout(pad=0)
            output_path = os.path.join(axis_dir, f'slice_{slice_count:04d}.png')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close()
            
            slice_count += 1
        
        print(f"Saved {slice_count} non-empty {axis_name} slices to {axis_dir}")

    print(f"All slices have been extracted and saved to: {output_dir}")

def main():
    img_path = "path/to/mri.img"
    output_dir = "sliced_mri"
    data = load_mri(img_path)
    extract_slices(data, output_dir)

if __name__ == "__main__":
    main()
>>>>>>> f0631fc6cf7aa9ff085a1e0c00e1b77a3c173394
