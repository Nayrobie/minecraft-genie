import requests
from bs4 import BeautifulSoup
import os
import json

"""
Scrapes specific pages and extracts clean text content.
What it does:
- Downloads HTML using requests
- Parses the HTML using BeautifulSoup to extract paragraph text
- Collects content from multiple pages
- Saves the results to a text file: lore_docs.txt

How to run:
    poetry run python data/web_scrapping.py
"""

WIKI_PAGES = {
    "trading": "https://minecraft.wiki/w/Trading",
    "brewing": "https://minecraft.wiki/w/Brewing",
    "enchanting": "https://minecraft.wiki/w/Enchanting",
    "mobs": "https://minecraft.wiki/w/Mob",
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
        str: The text content of the page including paragraphs and tables in original order.
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Remove edit links from HTML before processing
    for edit_link in soup.find_all(['span', 'a'], class_=['mw-editsection', 'mw-editsection-bracket']):
        edit_link.decompose()
    
    for unwanted in soup.find_all(['span'], class_=['mw-headline']):
        if unwanted.find('span', class_='mw-editsection'):
            unwanted.find('span', class_='mw-editsection').decompose()

    if DEBUG:
        print(soup.prettify()[:2000])  # Print first 2000 characters

    content_parts = []
    
    # Find all relevant elements in order (paragraphs, tables, headers, etc.)
    elements = soup.find_all(["p", "table", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li"])
    
    for element in elements:
        if element.name == "table":
            # Process table
            table_content = []
            rows = element.find_all("tr")
            for row in rows:
                cells = row.find_all(["th", "td"])
                if cells:
                    row_text = " | ".join(cell.get_text(strip=True) for cell in cells if cell.get_text(strip=True))
                    if row_text:
                        table_content.append(row_text)
            
            if table_content:
                content_parts.append("\nTABLE:\n" + "\n".join(table_content) + "\n")
        
        elif element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            # Process headers
            header_text = element.get_text(strip=True)
            if header_text:
                content_parts.append(f"\n{element.name.upper()}: {header_text}\n")
        
        elif element.name in ["ul", "ol"]:
            # Process lists
            list_items = element.find_all("li")
            if list_items:
                list_content = []
                for li in list_items:
                    li_text = li.get_text(strip=True)
                    if li_text:
                        list_content.append(f"- {li_text}")
                if list_content:
                    content_parts.append("\n".join(list_content))
        
        elif element.name == "p":
            # Process paragraphs
            text = element.get_text(strip=True)
            if text:
                content_parts.append(text)
    
    return "\n\n".join(content_parts)

def _clean_scraped_text(text: str) -> str:
    """
    Clean up scraped text by performing multiple cleaning operations:
    - Removes everything between "H2: Contents" and the next "H2:" header
    - Removes unwanted H2 sections (Video, History, Trivia, Gallery, Screenshots, References, Navigation)
    - Removes footnote reference letters after up arrow (↑abcd → ↑)
    - Cleans up excessive whitespace and line breaks
    
    Args:
        text (str): The scraped text content.
    Returns:
        str: Cleaned text with unwanted sections and formatting removed.
    """
    import re
    
    # Pattern to match "H2: Contents" followed by everything until the next "H2:"
    contents_pattern = r'H2: Contents.*?(?=H2: |\Z)'
    
    # Remove the contents sections (case insensitive)
    cleaned_text = re.sub(contents_pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove unwanted H2 sections and everything that follows until the next H2 or end
    unwanted_sections = ['Video', 'History', 'Trivia', 'Gallery', 'Screenshots', 'References', 'Navigation', 'Issues']
    
    for section in unwanted_sections:
        # Pattern to match "H2: SectionName" followed by everything until the next "H2:" or end
        section_pattern = rf'H2: {section}.*?(?=H2: |\Z)'
        cleaned_text = re.sub(section_pattern, '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove footnote reference letters after up arrow (↑abcd)
    # Pattern: ↑ followed by lowercase letters until we hit an uppercase letter or non-letter
    footnote_pattern = r'↑[a-z]+(?=[A-Z]|\s|[^a-zA-Z])'
    cleaned_text = re.sub(footnote_pattern, '↑', cleaned_text)
    
    # Clean up any multiple newlines left behind
    cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
    
    return cleaned_text.strip()

def _scrape_multiple_pages(pages: dict) -> dict:
    """
    Scrape multiple wiki pages and return their content.
    Args:
        pages (dict): A dictionary where keys are page names and values are URLs.
    Returns:
        dict: A dictionary with page names as keys and their text content as values.
    """
    all_content = {}
    for label, url in pages.items():
        print(f"➡️ Scraping {label} from {url}...")
        try:
            content = _scrape_page(url)
            content = _clean_scraped_text(content)
            all_content[label] = content
        except requests.RequestException as e:
            print(f"❌ Error scraping {label}: {e}")
            all_content[label] = ""
    print("✅ Scraping completed.")
    return all_content

# TODO: also save to txt file for human readability
def _save_to_json(content_dict: dict, path: str) -> None:
    """
    Save scraped wiki content as a structured JSON list.
    """
    data = []
    for page_name, content in content_dict.items():
        data.append({
            "title": page_name.capitalize(),
            "url": WIKI_PAGES[page_name],
            "content": content
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(data)} entries to {path}.")

if __name__ == "__main__":
    results = _scrape_multiple_pages(WIKI_PAGES)
    output_path = "data/lore.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _save_to_json(results, output_path)