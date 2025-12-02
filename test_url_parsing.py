import re

def test_url_parsing():
    urls = [
        "https://www.sofascore.com/gremio-fluminense/Iwcsob#id:12937512",
        "https://www.sofascore.com/football/match/gremio-fluminense/12937512",
        "12937512",
        "id:12937512"
    ]
    
    print("Testing URL Parsing:")
    for url in urls:
        match_id = None
        match_id_search = re.search(r'id:(\d+)', url)
        if match_id_search:
            match_id = match_id_search.group(1)
            print(f"✅ Regex match: {url} -> {match_id}")
        elif url.isdigit():
            match_id = url
            print(f"✅ Is digit: {url} -> {match_id}")
        else:
            # Try to find digits at the end of the URL
            # Common pattern: .../1234567
            # or .../slug/1234567
            last_part = url.split('/')[-1]
            if last_part.isdigit():
                match_id = last_part
                print(f"✅ Last part digit: {url} -> {match_id}")
            else:
                print(f"❌ Failed: {url}")

if __name__ == "__main__":
    test_url_parsing()
