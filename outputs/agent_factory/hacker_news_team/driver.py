import json
import sys
from tools import get_hn_top_stories, get_story_details

def main():
    command = sys.argv[1]
    
    if command == "fetch_top":
        stories = get_hn_top_stories(limit=30)
        print(json.dumps(stories, indent=2))
        
    elif command == "fetch_details":
        item_id = int(sys.argv[2])
        details = get_story_details(item_id)
        print(json.dumps(details, indent=2))

if __name__ == "__main__":
    main()
