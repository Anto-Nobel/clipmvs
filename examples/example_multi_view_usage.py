from PIL import Image
from clipmvs.multi_view_summarizer import MultiViewSummarizer

# Initialize the summarizer
summarizer = MultiViewSummarizer(config_path='config.json')

# Process videos to store embeddings
video_paths = ['path_to_your_video1.mp4', 'path_to_your_video2.mp4']  # Add more video paths as needed
summarizer.process_videos(video_paths)

# Define multiple queries (both text and image)
queries = [
    "A person playing guitar",
    Image.open('path_to_query_image1.jpg'),
    "A cat sitting on a sofa",
    Image.open('path_to_query_image2.jpg')
]
is_images = [False, True, False, True]

# Generate and visualize summaries for multiple queries
summarizer.generate_and_visualize_summaries(queries, video_paths[0], top_k=5, is_images=is_images)

# Close the Qdrant connection
summarizer.close()
