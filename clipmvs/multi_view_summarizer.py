import csv
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from .video_loader import VideoDataLoader
from .clip_retriever import CLIPEmbeddingRetriever
from .qdrant_handler import QdrantHandler
from PIL import Image

class MultiViewSummarizer:
    """
    A class to handle multi-view summarization of videos using CLIP and Qdrant.
    """
    def __init__(self, config_path='config.json'):
        """
        Initialize the MultiViewSummarizer.

        Args:
            config_path (str): Path to the configuration JSON file.
        """
        self.retriever = CLIPEmbeddingRetriever()
        self.qdrant_handler = QdrantHandler(config_path)

    def process_videos(self, video_paths, batch_size=5, interval=10):
        """
        Process multiple videos concurrently to extract and store CLIP embeddings.

        Args:
            video_paths (list): List of paths to the video files.
            batch_size (int): Number of frames to fetch in each batch.
            interval (int): Interval between frames to fetch.
        """
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_single_video, video_path, batch_size, interval) for video_path in video_paths]
            for future in futures:
                future.result()  # Wait for all futures to complete

    def process_single_video(self, video_path, batch_size, interval):
        """
        Process a single video to extract and store CLIP embeddings.

        Args:
            video_path (str): Path to the video file.
            batch_size (int): Number of frames to fetch in each batch.
            interval (int): Interval between frames to fetch.
        """
        video_loader = VideoDataLoader(video_path, batch_size, interval=interval)
        for frames, timestamps in video_loader:
            image_embeddings = self.retriever.get_CLIP_vision_embedding(frames)
            metadata = [{"timestamp": ts} for ts in timestamps]
            self.qdrant_handler.store_embedding(image_embeddings, metadata)

    def generate_summary(self, query, video_path, top_k=5, is_image=False):
        """
        Generate a multi-view summary for a given query.

        Args:
            query (str or PIL.Image.Image): The query text or image.
            video_path (str): Path to the video file for visualization.
            top_k (int): The number of top results to retrieve.
            is_image (bool): Flag to indicate if the query is an image.

        Returns:
            list: A summary combining textual and visual information.
        """
        if is_image:
            query_embedding = self.retriever.get_CLIP_vision_embedding([query])[0]
        else:
            query_embedding = self.retriever.get_CLIP_text_embedding(query)

        results = self.qdrant_handler.query_embedding(query_embedding, top_k=top_k)
        
        # Extract visual information
        video_loader = VideoDataLoader(video_path)
        summary = []
        for result in results:
            timestamp = result.payload['timestamp']
            frame = video_loader.get_frame_by_timestamp(timestamp)
            summary.append({
                "timestamp": timestamp,
                "frame": frame,
                "similarity": result.score
            })
        
        return summary

    def generate_and_visualize_summaries(self, queries, video_path, top_k=5, is_images=None):
        """
        Generate and visualize summaries for multiple queries.

        Args:
            queries (list): List of queries (texts or images).
            video_path (str): Path to the video file for visualization.
            top_k (int): The number of top results to retrieve for each query.
            is_images (list): List of boolean values indicating if each query is an image.
        """
        if is_images is None:
            is_images = [False] * len(queries)
        
        all_summaries = []
        for i, query in enumerate(queries):
            summary = self.generate_summary(query, video_path, top_k=top_k, is_image=is_images[i])
            all_summaries.append(summary)
            self.visualize_summary(summary, video_path, query, is_images[i])
            csv_path = f'summary_{i}.csv'  # Create a CSV for each query
            self.save_summary_to_csv(summary, csv_path)

    def visualize_summary(self, summary, video_path, query, is_image):
        """
        Visualize the summary generated.

        Args:
            summary (list): The summary data containing timestamps and frames.
            video_path (str): Path to the video file for visualization.
            query (str or PIL.Image.Image): The query text or image.
            is_image (bool): Flag to indicate if the query is an image.
        """
        # Visualize frames
        plt.figure(figsize=(15, 5))
        for i, item in enumerate(summary):
            frame = item["frame"]
            timestamp = item["timestamp"]
            if frame:
                plt.subplot(1, len(summary), i + 1)
                plt.imshow(frame)
                plt.title(f"Timestamp: {timestamp:.2f}s\nScore: {item['similarity']:.4f}")
                plt.axis('off')
        plt.suptitle(f'Summary for query: {"Image" if is_image else query}')
        plt.show()

        # Visualize timeline
        timestamps = [item["timestamp"] for item in summary]
        scores = [item["similarity"] for item in summary]

        plt.figure(figsize=(10, 2))
        plt.hlines(1, 0, max(timestamps), colors='gray', linestyles='dashed')
        plt.eventplot(timestamps, orientation='horizontal', colors='blue')
        for ts, score in zip(timestamps, scores):
            plt.text(ts, 1.05, f"{ts:.2f}s\n{score:.4f}", ha='center', va='bottom', fontsize=8)
        plt.title(f'Queried Vectors on Timeline for query: {"Image" if is_image else query}')
        plt.xlabel('Timestamp (s)')
        plt.yticks([])
        plt.show()

    def save_summary_to_csv(self, summary, csv_path):
        """
        Save the summary metadata to a CSV file.

        Args:
            summary (list): The summary data containing timestamps and frames.
            csv_path (str): Path to the output CSV file.
        """
        fieldnames = ['timestamp', 'similarity']
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for item in summary:
                writer.writerow({'timestamp': item['timestamp'], 'similarity': item['similarity']})

    def close(self):
        """
        Close the Qdrant connection.
        """
        self.qdrant_handler.close()
