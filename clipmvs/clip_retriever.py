import torch
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection, AutoProcessor, AutoTokenizer
import matplotlib.pyplot as plt

class CLIPEmbeddingRetriever:
    """
    A class to retrieve CLIP embeddings for text and images.
    """
    def __init__(self):
        """
        Initialize the CLIPEmbeddingRetriever.
        """
        self.vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to("cuda:0")
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to("cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
    @torch.no_grad()
    def get_CLIP_text_embedding(self, searchword):
        """
        Get the CLIP embedding for a search word.

        Args:
            searchword (str): The search word to embed.

        Returns:
            numpy.ndarray: The text embedding.
        """
        text_inputs = self.tokenizer(searchword, padding=True, return_tensors="pt").to("cuda:0")
        text_outputs = self.text_model(**text_inputs)
        text_embeds = text_outputs.text_embeds.cpu().detach().numpy()
        return text_embeds
    
    @torch.no_grad()
    def get_CLIP_vision_embedding(self, images):
        """
        Get the CLIP embedding for a list of images.

        Args:
            images (list): List of PIL.Image objects.

        Returns:
            numpy.ndarray: The image embeddings.
        """
        image_inputs = self.processor(images=images, return_tensors="pt").to("cuda:0")
        image_outputs = self.vision_model(**image_inputs)
        image_embeds = image_outputs.image_embeds.cpu().detach().numpy()
        return image_embeds

    def visualize_retrieved_frames(self, video_loader, results):
        """
        Visualize the frames retrieved from a query.

        Args:
            video_loader (VideoDataLoader): The video loader to fetch frames.
            results (list): List of query results.
        """
        timestamps = [point.payload['timestamp'] for point in results]
        frames = [video_loader.get_frame_by_timestamp(ts) for ts in timestamps]

        plt.figure(figsize=(15, 5))
        for i, frame in enumerate(frames):
            if frame:
                plt.subplot(1, len(frames), i + 1)
                plt.imshow(frame)
                plt.title(f"Timestamp: {timestamps[i]:.2f}s")
                plt.axis('off')
        plt.show()
