import requests
from bs4 import BeautifulSoup
import os
import json
from urllib.parse import urlparse
import re
from typing import Optional, List
from bs4 import Tag

# TODO: finish gold examples
# create an eval.py script to test the examples
# test with current version
# stash pop with table embedding code enhancement
# run the eval again to compare

"""
Scrape specific pages and extract clean text content.
What it does:
- Downloads HTML using requests
- Parses the HTML using BeautifulSoup to extract paragraph text
- Collects content from multiple pages
- Saves the results to a text file: lore_docs.txt

How to run:
    poetry run python data/web_scraping.py
"""

WIKI_PAGES = {
    "trading": "https://minecraft.wiki/w/Trading",
    "brewing": "https://minecraft.wiki/w/Brewing",
    "enchanting": "https://minecraft.wiki/w/Enchanting",
    "mobs": "https://minecraft.wiki/w/Mob",
    "blocks": "https://minecraft.wiki/w/Block",
    "items": "https://minecraft.wiki/w/Item",
    "crafting information": "https://minecraft.wiki/w/Crafting",
    "crafting recipes": "https://www.minecraftcrafting.info",
    "smelting": "https://minecraft.wiki/w/Smelting",
    "tutorials": "https://minecraft.wiki/w/Tutorials",  # TODO: tutorials are only links, make sure the tool can provide the links
    "redstone": "https://minecraft.wiki/w/Redstone_circuits",
}

DEBUG = True

def _extract_intro_before_first_table(soup: BeautifulSoup) -> str:
    """
    Extract only the introductory paragraphs that occur before the first table.

    Args:
        soup (BeautifulSoup): Parsed HTML document.

    Return:
        str: Intro text before the first table (collapsed), or empty string.
    """
    # Choose a broad main container
    main: Tag | None = soup.select_one("main, .mw-parser-output, #content, body")
    if not main:
        return ""

    first_table: Optional[Tag] = main.find("table")
    if not first_table:
        # No tables; just return top paragraphs
        paras: List[str] = []
        for p in main.find_all("p"):
            txt = p.get_text(" ", strip=True)
            if txt:
                paras.append(txt)
        return re.sub(r"\s{2,}", " ", " ".join(paras)).strip()

    # Collect text from siblings that appear BEFORE the first table
    paras: List[str] = []
    for el in list(main.children):
        if isinstance(el, Tag) and el.name == "table":
            break
        if isinstance(el, Tag) and el.name in {"p", "div"}:
            txt = el.get_text(" ", strip=True)
            if txt:
                paras.append(txt)
    intro: str = " ".join(paras)
    return re.sub(r"\s{2,}", " ", intro).strip()

def _is_minecraftcrafting(url: str) -> bool:
    """
    Check whether a URL belongs to minecraftcrafting.info.

    Args:
        url (str): Absolute URL.

    Return:
        bool: True if host is minecraftcrafting.info, else False.
    """
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return "minecraftcrafting.info" in host


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

    is_crafting_site = _is_minecraftcrafting(url)

    # Remove edit links from HTML before processing
    for edit_link in soup.find_all(
        ["span", "a"], class_=["mw-editsection", "mw-editsection-bracket"]
    ):
        edit_link.decompose()

    for unwanted in soup.find_all(["span"], class_=["mw-headline"]):
        if unwanted.find("span", class_="mw-editsection"):
            unwanted.find("span", class_="mw-editsection").decompose()

    if DEBUG:
        print(soup.prettify()[:2000])  # Print first 2000 characters

    content_parts = []
    # Determine if this is the tutorials page
    is_tutorials_page = url == WIKI_PAGES["tutorials"]

    # Find all relevant elements in order (paragraphs, tables, headers, etc.)
    elements = soup.find_all(
        ["p", "table", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "div"]
    )

    icon_classes = ["icon", "mob-icon"]

    # Special handling for minecraftcrafting.info: capture only the intro text before the first table once.
    if is_crafting_site:
        intro_text = _extract_intro_before_first_table(soup)
        if intro_text:
            content_parts.append(intro_text)

    for element in elements:
        # Extracts image/icon labels inline if this is an icon container
        if element.name == "div" and any(
            cls in element.get("class", []) for cls in icon_classes
        ):
            name_div = element.find_next_sibling(
                "div", class_="name"
            ) or element.find_next_sibling("div", class_="mob-name")
            if name_div:
                img_name = name_div.get_text(strip=True)
                if img_name:
                    content_parts.append(f"IMAGE_LABEL: {img_name}")
                continue

            name_child = element.find("div", class_="name") or element.find(
                "div", class_="mob-name"
            )
            if name_child:
                img_name = name_child.get_text(strip=True)
                if img_name:
                    content_parts.append(f"IMAGE_LABEL: {img_name}")
                continue

            img_tag = element.find("img")
            if img_tag:
                alt_text = img_tag.get("alt")
                title_text = img_tag.get("title")
                label = alt_text or title_text
                if label:
                    content_parts.append(f"IMAGE_LABEL: {label}")
                    continue

            possible_name = element.get_text(strip=True)
            if possible_name:
                content_parts.append(f"IMAGE_LABEL: {possible_name}")
            continue

        # Process table, headers, lists, and paragraphs
        if element.name == "table":
            # Process table
            table_content = []
            rows = element.find_all("tr")
            for row in rows:
                cells = row.find_all(["th", "td"])
                if cells:
                    row_text = " | ".join(
                        cell.get_text(strip=True)
                        for cell in cells
                        if cell.get_text(strip=True)
                    )
                    if row_text:
                        table_content.append(row_text)

            if table_content:
                content_parts.append("\nTABLE:\n" + "\n".join(table_content) + "\n")

        elif element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            # Process headers
            header_text = element.get_text(strip=True)
            if header_text:
                content_parts.append(f"{element.name.upper()}: {header_text}")

        elif element.name in ["ul", "ol"]:
            # Process lists
            list_items = element.find_all("li")
            if list_items:
                list_content = []
                for li in list_items:
                    # Only extract links for tutorials page
                    if is_tutorials_page:
                        a_tag = li.find("a")
                        if a_tag and a_tag.get("href"):
                            link_text = a_tag.get_text(" ", strip=True)
                            href = a_tag.get("href")
                            if href.startswith("/"):
                                href = f"https://minecraft.wiki{href}"
                            list_content.append(f"- [{link_text}]({href})")
                        else:
                            li_text = li.get_text(" ", strip=True)
                            if li_text:
                                list_content.append(f"- {li_text}")
                    else:
                        # For other pages, just output text
                        li_text = li.get_text(" ", strip=True)
                        if li_text:
                            list_content.append(f"- {li_text}")
                if list_content:
                    content_parts.append("\n".join(list_content))

        elif element.name == "p":
            # Process paragraphs
            # For minecraftcrafting.info we will add only the intro before the first table,
            # so skip generic paragraph capture entirely to avoid duplicating table text into a single long line.
            if is_crafting_site:
                continue
            text = element.get_text(" ", strip=True)
            if text:
                content_parts.append(text)

    return "\n".join(content_parts)


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

    # Clean unwanted section
    contents_pattern = r"H2: Contents.*?(?=H2: |\Z)"
    cleaned_text = re.sub(contents_pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    unwanted_h2_sections = [
        "Data values",
        "Sounds",
        "Video",
        "History",
        "Issues",
        "Gallery",
        "See also",
        "Screenshots",
        "References",
        "External links",
        "Navigation",
        "Changed recipes",
        "Complete recipe list",
    ]

    for section in unwanted_h2_sections:
        section_pattern = rf"H2: {section}.*?(?=H2: |\Z)"
        cleaned_text = re.sub(
            section_pattern, "", cleaned_text, flags=re.DOTALL | re.IGNORECASE
        )
    # Remove unwanted H3 sections
    unwanted_h3_sections = [
        "Unused mobs",
        "Education mobs",
        "Removed mobs",
        "Joke mobs",
        "Unimplemented mobs",
        "Mentioned mobs",
        "Education blocks",
        "Removed blocks",
        "Joke blocks",
    ]
    for section in unwanted_h3_sections:
        section_pattern = rf"H3: {section}.*?(?=H3: |H2: |\Z)"
        cleaned_text = re.sub(
            section_pattern, "", cleaned_text, flags=re.DOTALL | re.IGNORECASE
        )

    # Remove footnote reference letters after up arrow (↑abcd)
    footnote_pattern = r"↑[a-z]+(?=[A-Z]|\s|[^a-zA-Z])"
    cleaned_text = re.sub(footnote_pattern, "↑", cleaned_text)

    # Remove elements from crafting recipes website
    crafting_banner = "BasicBlocksToolsDefenceMechanismFoodOtherDyeWoolBrewing"
    if crafting_banner in cleaned_text:
        cleaned_text = cleaned_text.replace(crafting_banner, "")
        cleaned_text = cleaned_text.replace(" | Image", "")
        cleaned_text = cleaned_text.replace("[Back to top]", "")
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


def _save_to_txt(content_dict: dict, path: str) -> None:
    """
    Save scraped wiki content as a human-readable text file.
    """
    with open(path, "w", encoding="utf-8") as f:
        for page_name, content in content_dict.items():
            f.write(f"{'=' * 40}\n")
            f.write(f"PAGE NAME: {page_name.upper()}\n")
            f.write(f"URL: {WIKI_PAGES[page_name]}\n")
            f.write(f"{'=' * 40}\n")
            f.write(content)
            f.write("\n\n")
    print(f"✅ Saved {len(content_dict)} entries to {path}.")


def _save_to_json(content_dict: dict, path: str) -> None:
    """
    Save scraped wiki content as a structured JSON list.
    """
    data = []
    for page_name, content in content_dict.items():
        data.append(
            {
                "title": page_name.capitalize(),
                "url": WIKI_PAGES[page_name],
                "content": content,
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(data)} entries to {path}.")


if __name__ == "__main__":
    results = _scrape_multiple_pages(WIKI_PAGES)

    json_output_path = "data/lore.json"
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
    _save_to_json(results, json_output_path)

    txt_output_path = "data/lore.txt"
    _save_to_txt(results, txt_output_path)
