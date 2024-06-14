import os
from concurrent.futures import ThreadPoolExecutor
from clip_video_processor import VideoDataLoader, CLIPEmbeddingRetriever, QdrantHandler
from PIL import Image

def process_video(video_path, retriever, qdrant_handler, batch_size=5, interval=10):
    """
    Process a single video file to extract and store CLIP embeddings.

    Args:
        video_path (str): Path to the video file.
        retriever (CLIPEmbeddingRetriever): The CLIP retriever instance.
        qdrant_handler (QdrantHandler): The Qdrant handler instance.
        batch_size (int): Number of frames to fetch in each batch.
        interval (int): Interval between frames to fetch.
    """
    video_loader = VideoDataLoader(video_path, batch_size, interval=interval)
    for frames, timestamps in video_loader:
        image_embeddings = retriever.get_CLIP_vision_embedding(frames)
        metadata = [{"timestamp": ts} for ts in timestamps]
        qdrant_handler.store_embedding(image_embeddings, metadata)

# Example usage:
video_paths = ['path_to_your_video1.mp4', 'path_to_your_video2.mp4']  # Add more video paths as needed
retriever = CLIPEmbeddingRetriever()
qdrant_handler = QdrantHandler(config_path='config.json')  # Specify the path to your config file

# Use ThreadPoolExecutor to process multiple videos concurrently
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_video, video_path, retriever, qdrant_handler) for video_path in video_paths]
    for future in futures:
        future.result()  # Wait for all futures to complete

# Query example:
sample_frame = Image.open('path_to_sample_frame.jpg')  # Replace with your sample frame
sample_embedding = retriever.get_CLIP_vision_embedding([sample_frame])
results = qdrant_handler.query_embedding(sample_embedding)
print("Query Results:", results)

# Visualize the retrieved frames
video_loader = VideoDataLoader(video_paths[0])  # You can change this to the relevant video path
retriever.visualize_retrieved_frames(video_loader, results)

# Close the Qdrant connection
qdrant_handler.close()
