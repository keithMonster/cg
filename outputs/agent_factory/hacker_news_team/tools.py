import requests
import json
from typing import List, Dict, Any

def get_hn_top_stories(limit: int = 30) -> List[Dict[str, Any]]:
    """
    Fetches the top stories from Hacker News.
    
    Args:
        limit (int): The number of stories to fetch. Defaults to 30.
        
    Returns:
        List[Dict[str, Any]]: A list of story objects (title, url, score, id).
    """
    base_url = "https://hacker-news.firebaseio.com/v0"
    try:
        top_ids = requests.get(f"{base_url}/topstories.json").json()[:limit]
        stories = []
        for item_id in top_ids:
            story = requests.get(f"{base_url}/item/{item_id}.json").json()
            if story and 'title' in story: # Basic validation
                stories.append({
                    "id": story.get("id"),
                    "title": story.get("title"),
                    "url": story.get("url", ""),
                    "score": story.get("score", 0),
                    "descendants": story.get("descendants", 0), # Comment count
                    "time": story.get("time")
                })
        return stories
    except Exception as e:
        print(f"Error fetching HN stories: {e}")
        return []

def get_story_details(item_id: int) -> Dict[str, Any]:
    """
    Fetches details for a specific HN item, including top comments.
    
    Args:
        item_id (int): The ID of the story.
        
    Returns:
        Dict[str, Any]: The full story object with a 'comments' field containing top-level comment texts.
    """
    base_url = "https://hacker-news.firebaseio.com/v0"
    try:
        story = requests.get(f"{base_url}/item/{item_id}.json").json()
        comments = []
        if 'kids' in story:
            for kid_id in story['kids'][:5]: # Limit to top 5 comments for efficiency
                comment = requests.get(f"{base_url}/item/{kid_id}.json").json()
                if comment and 'text' in comment:
                    comments.append(comment['text'])
        story['comments'] = comments
        return story
    except Exception as e:
        print(f"Error fetching story details: {e}")
        return {}

def write_report(filename: str, content: str) -> str:
    """
    Writes the final report to a markdown file.
    
    Args:
        filename (str): The name of the file (e.g., 'daily_report_2025_11_28.md').
        content (str): The markdown content.
        
    Returns:
        str: Success message or error.
    """
    try:
        with open(filename, 'w') as f:
            f.write(content)
        return f"Successfully wrote report to {filename}"
    except Exception as e:
        return f"Error writing file: {e}"
