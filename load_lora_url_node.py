import os
import hashlib
import requests
import tarfile
import tempfile
from urllib.parse import urlparse
import folder_paths
import comfy.utils
import comfy.sd
from tqdm import tqdm

class LoadLoraFromURL:
    """Load a LoRA model from a URL"""
    
    def __init__(self):
        self.cache_dir = os.path.join(folder_paths.get_input_directory(), "url_loras")
        os.makedirs(self.cache_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {"default": ""}),
                "model": ("MODEL",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "loaders"

    def download_if_needed(self, url):
        """Download the file or extract from a .tar if needed and return a local .safetensors path"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        url_path = urlparse(url).path.lower()

        is_tar = url_path.endswith('.tar') or url_path.endswith('.tar.gz') or url_path.endswith('.tgz')

        # Cache target is always a .safetensors file for loading
        safetensors_filename = f"{url_hash}.safetensors" if not is_tar else f"{url_hash}_flux-lora.safetensors"
        safetensors_local_path = os.path.join(self.cache_dir, safetensors_filename)

        if os.path.exists(safetensors_local_path):
            return safetensors_local_path

        print(f"Downloading LoRA from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        if not is_tar:
            # Direct .safetensors download
            with open(safetensors_local_path, 'wb') as f, tqdm(
                desc=safetensors_filename,
                total=total_size,
                unit='iB',
                unit_scale=True
            ) as pbar:
                for data in response.iter_content(chunk_size=8192):
                    size = f.write(data)
                    pbar.update(size)
            return safetensors_local_path

        # .tar download and selective extract of flux-lora/flux-lora.safetensors
        with tempfile.NamedTemporaryFile(delete=False) as tmp_archive:
            tmp_archive_path = tmp_archive.name
            with tqdm(
                desc=os.path.basename(urlparse(url).path) or f"{url_hash}.tar",
                total=total_size,
                unit='iB',
                unit_scale=True
            ) as pbar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        tmp_archive.write(chunk)
                        pbar.update(len(chunk))

        try:
            # Open tar and find target file
            with tarfile.open(tmp_archive_path, 'r:*') as tar:
                members = tar.getmembers()
                # Prefer exact path
                target_member = None
                for m in members:
                    if m.isfile() and m.name.replace('\\', '/').endswith('flux-lora/flux-lora.safetensors'):
                        target_member = m
                        break
                # Fallback: any .safetensors inside a directory named flux-lora
                if target_member is None:
                    for m in members:
                        if not m.isfile():
                            continue
                        normalized = m.name.replace('\\', '/')
                        parts = normalized.split('/')
                        if any(part == 'flux-lora' for part in parts[:-1]) and parts[-1].endswith('.safetensors'):
                            target_member = m
                            break

                if target_member is None:
                    raise FileNotFoundError("Target 'flux-lora/flux-lora.safetensors' not found in archive")

                # Extract only the target file into the cache path
                extracted_file = tar.extractfile(target_member)
                if extracted_file is None:
                    raise FileNotFoundError("Unable to extract target file from archive")

                with open(safetensors_local_path, 'wb') as out_f:
                    while True:
                        chunk = extracted_file.read(1024 * 1024)
                        if not chunk:
                            break
                        out_f.write(chunk)

            return safetensors_local_path
        finally:
            try:
                os.remove(tmp_archive_path)
            except Exception:
                # Best-effort cleanup
                pass

    def load_lora(self, url, model, strength):
        try:
            # Download or get cached file
            lora_path = self.download_if_needed(url)
            
            # Load the LoRA using ComfyUI's built-in functions
            lora = comfy.utils.load_torch_file(lora_path)
            model_lora, _ = comfy.sd.load_lora_for_models(
                model, None, lora, strength, 0
            )
            return (model_lora,)
            
        except Exception as e:
            print(f"Error loading LoRA: {str(e)}")
            return (model,)

# ComfyUI node display name mapping
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadLoraFromURL": "Load Flux LoRA from URL/TAR",
}