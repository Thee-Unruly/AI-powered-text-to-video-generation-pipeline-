import torch
from diffusers import StableVideoDiffusionPipeline

# Set up the environment & load model with GPU acceleration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion", device=device)

# Load the video file
def generate_video(scene_data):
    with torch.inference_model():
        video = pipe(scene_data).videos[0] # generate video
    video.save('output.mp4')
    
    return "output.mp4"

# No paid APIs
# Uses GPU acceleration (CUDA support)
# Stable & Open-source