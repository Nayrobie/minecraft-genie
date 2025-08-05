import requests
from bs4 import BeautifulSoup
import os

"""
Scrapes specific pages and extracts clean text content.
What it does:
- Downloads HTML using requests
- Parses the HTML using BeautifulSoup to extract paragraph text
- Collects content from multiple pages
- Saves the results to a text file: lore_docs.txt

How to run:
    poetry run python data/web_scraping.py
"""

WIKI_PAGES = {
    "mobs": "https://minecraft.wiki/w/Mob",
    "trading": "https://minecraft.wiki/w/Trading",
    "brewing": "https://minecraft.wiki/w/Brewing",
    "enchanting": "https://minecraft.wiki/w/Enchanting",
    "blocks": "https://minecraft.wiki/w/Block",
    "items": "https://minecraft.wiki/w/Item",
    "crafting": "https://minecraft.wiki/w/Crafting",
    "smelting": "https://minecraft.wiki/w/Smelting",
    "tutorials": "https://minecraft.wiki/w/Tutorials",
    "redstone": "https://minecraft.wiki/w/Redstone_circuits",
}

DEBUG = True

def _scrape_page(url: str) -> str:
    """
    Scrape the content of a wiki page and return it as a string.
    Args:
        url (str): The URL of the wiki page to scrape.
    Returns:
        str: The text content of the page.
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    if DEBUG:
        print(soup.prettify()[:2000])  # Print first 2000 characters for debugging

    paragraphs = soup.find_all("p")
    content = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    return content

def _scrape_multiple_pages(pages: dict) -> dict:
    """
    Scrape multiple wiki pages and return their content.
    Args:
        pages (dict): A dictionary where keys are page names and values are URLs.
    Returns:
        dict: A dictionary with page names as keys and their text content as values.
    """
    all_texts = []
    for label, url in pages.items():
        print(f"➡️ Scraping {label} from {url}...")
        try:
            content = _scrape_page(url)
            all_texts.append(content)
        except requests.RequestException as e:
            print(f"❌ Error scraping {label}: {e}")
    print("✅ Scraping completed.")
    return all_texts

def _save_to_file(texts: list[str], path: str) -> None:
    """
    Saves a list of text chunks to a file, separated by double newlines.

    Args:
        texts (List[str]): The list of string contents to save.
        path (str): Path to the output file.
    """
    with open(path, "w", encoding="utf-8") as f:
        for chunk in texts:
            f.write(chunk + "\n\n")
    print(f"✅ Saved {len(texts)} chunks to {path}.")

if __name__ == "__main__":
    results = _scrape_multiple_pages(WIKI_PAGES)
    output_path = "data/lore_docs.txt"
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    _save_to_file(results, output_path)