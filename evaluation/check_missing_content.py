"""
Check which gold prompt snippets are missing from the scraped data.
This helps identify what needs to be added to the web scraping.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
GOLD_PROMPTS_PATH = PROJECT_ROOT / "evaluation/gold_prompts.json"
LORE_PATH = PROJECT_ROOT / "data/lore.json"


def load_json(path: Path) -> List[Dict]:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_for_search(text: str) -> str:
    """Normalize text for more flexible searching."""
    # Convert to lowercase
    text = text.lower()
    # Remove special Unicode characters
    text = text.replace("\u200c", "")  # Zero-width non-joiner
    text = text.replace("\u200b", "")  # Zero-width space
    text = text.replace("‌", "")
    # Normalize fractions
    text = text.replace(" ⁄ ", "/")
    text = text.replace("⁄", "/")
    return text


def find_snippet_in_content(snippet: str, content: str, context_chars: int = 100) -> Tuple[bool, str]:
    """
    Search for snippet in content with normalization.
    Returns (found, context) tuple.
    """
    norm_snippet = normalize_for_search(snippet)
    norm_content = normalize_for_search(content)
    
    if norm_snippet in norm_content:
        # Find position in normalized content
        idx = norm_content.find(norm_snippet)
        # Get context from original content (approximate position)
        start = max(0, idx - context_chars)
        end = min(len(content), idx + len(snippet) + context_chars)
        context = content[start:end].replace('\n', ' ')
        return True, f"...{context}..."
    
    return False, ""


def check_missing_content():
    """Main function to check missing content."""
    print("=" * 80)
    print("MISSING CONTENT ANALYSIS")
    print("=" * 80)
    
    # Load data
    gold_prompts = load_json(GOLD_PROMPTS_PATH)
    lore_data = load_json(LORE_PATH)
    
    # Combine all content for searching
    all_content = "\n\n".join([entry["content"] for entry in lore_data])
    
    # Track statistics
    total_questions = len(gold_prompts)
    total_snippets = sum(len(q["expected_answer_contains"]) for q in gold_prompts)
    found_snippets = 0
    missing_snippets = []
    questions_with_missing = []
    
    print(f"\nAnalyzing {total_questions} questions with {total_snippets} expected snippets...\n")
    
    # Check each question
    for i, prompt in enumerate(gold_prompts, 1):
        question = prompt["question"]
        expected_snippets = prompt["expected_answer_contains"]
        source_link = prompt.get("source_link", "N/A")
        
        # Check each snippet
        missing_for_question = []
        for snippet in expected_snippets:
            found, context = find_snippet_in_content(snippet, all_content)
            
            if found:
                found_snippets += 1
            else:
                missing_for_question.append(snippet)
                missing_snippets.append({
                    "question": question,
                    "snippet": snippet,
                    "source_link": source_link,
                })
        
        # Report if any snippets missing
        if missing_for_question:
            questions_with_missing.append(question)
            print(f"❌ Question {i}: {question}")
            print(f"   Source: {source_link}")
            for snippet in missing_for_question:
                print(f"   Missing: '{snippet}'")
            print()
    
    # Summary statistics
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total questions: {total_questions}")
    print(f"Questions with all snippets found: {total_questions - len(questions_with_missing)}")
    print(f"Questions with missing snippets: {len(questions_with_missing)}")
    print(f"\nTotal snippets: {total_snippets}")
    print(f"Snippets found: {found_snippets} ({found_snippets/total_snippets*100:.1f}%)")
    print(f"Snippets missing: {len(missing_snippets)} ({len(missing_snippets)/total_snippets*100:.1f}%)")
    
    # Group missing snippets by source
    print("\n" + "=" * 80)
    print("MISSING CONTENT BY SOURCE")
    print("=" * 80)
    
    by_source = {}
    for item in missing_snippets:
        source = item["source_link"]
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(item)
    
    for source, items in sorted(by_source.items()):
        print(f"\n📄 {source}")
        print(f"   {len(items)} missing snippets:")
        for item in items:
            print(f"   - '{item['snippet']}' (Q: {item['question'][:60]}...)")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if len(missing_snippets) == 0:
        print("✅ All expected snippets found in scraped data!")
        print("   Focus on improving chunking and embedding strategy.")
    else:
        print("🔧 Web scraping improvements needed:")
        print("\n1. Add or improve scraping for these pages:")
        for source in sorted(set(item["source_link"] for item in missing_snippets)):
            print(f"   - {source}")
        
        print("\n2. Specific content to capture:")
        # Group by type
        urls = [item for item in missing_snippets if item["snippet"].startswith("http")]
        terms = [item for item in missing_snippets if not item["snippet"].startswith("http")]
        
        if urls:
            print(f"\n   URLs ({len(urls)} missing):")
            for item in urls[:5]:  # Show first 5
                print(f"   - {item['snippet']}")
            if len(urls) > 5:
                print(f"   ... and {len(urls) - 5} more")
        
        if terms:
            print(f"\n   Terms/Concepts ({len(terms)} missing):")
            unique_terms = list(set(item["snippet"] for item in terms))
            for term in sorted(unique_terms)[:10]:  # Show first 10
                print(f"   - '{term}'")
            if len(unique_terms) > 10:
                print(f"   ... and {len(unique_terms) - 10} more")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    check_missing_content()
