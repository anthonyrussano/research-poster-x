#!/usr/bin/env python3
"""
End-to-end helper:
1) Scrape a random natural-healing article from wikip.co.
2) Ask the local LM Studio model to craft an engaging X post.
3) Push the post to the X Android app via uiautomator2.

Environment variables expected:
- LMSTUDIO_BASE_URL   (e.g. http://localhost:1234/v1)
- LMSTUDIO_MODEL      (model name served by LM Studio)
- OPENAI_API_KEY      (optional; defaults to 'lm-studio' if unset)
- GALAXY_IP           (device serial/IP for uiautomator2 connect)
"""

import argparse
import os
import random
import time
from typing import List, Optional, Tuple

import requests
import uiautomator2 as u2
from bs4 import BeautifulSoup
from openai import OpenAI

try:
    from dotenv import load_dotenv

    if os.path.exists(".env"):
        load_dotenv()
except ImportError:
    # Optional dependency; skip if not installed
    pass


BASE_URL = "https://wikip.co"
CATEGORY_PATH = "/categories/natural-healing/"
LM_BASE_URL = os.getenv("LMSTUDIO_BASE_URL")
LM_MODEL = os.getenv("LMSTUDIO_MODEL")
LM_API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("API_KEY", "lm-studio"))
ANDROID_SERIAL = os.getenv("GALAXY_IP")


def fetch_random_article(
    base_url: str, category_path: str, attempted: Optional[List[str]] = None
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Return (title, content, image_url, article_url) or Nones on failure."""
    attempted = attempted or []
    url = f"{base_url}{category_path}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    links = soup.find_all("a", class_="timeline-article-title")
    links = [link for link in links if base_url + link.get("href") not in attempted]
    if not links:
        return None, None, None, "No suitable links found."

    random_link = random.choice(links)
    article_url = base_url + random_link.get("href")
    attempted.append(article_url)

    article_resp = requests.get(article_url, timeout=20)
    article_resp.raise_for_status()
    article_soup = BeautifulSoup(article_resp.text, "html.parser")

    title_tag = article_soup.find("h1", class_="article-title")
    title = title_tag.find("span").text.strip() if title_tag and title_tag.find("span") else None

    image_tag = article_soup.find("img")
    image_url = image_tag["src"] if image_tag else None

    # Drop footnotes so they do not pollute the prompt
    footnotes_section = article_soup.find("section", class_="footnotes")
    if footnotes_section:
        footnotes_section.decompose()

    content_list: List[str] = []
    h2_tag = article_soup.find(
        "h2",
        id=lambda x: x in ["Healing-Properties", "Biological-Properties", "Disease-Symptom-Treatment"],
    )
    if h2_tag:
        h3_elements = h2_tag.find_all_next("h3", limit=10)
        for h3 in h3_elements:
            content_list.append(h3.text.strip())
            for sibling in h3.next_siblings:
                if sibling.name in ["h3", "h2"]:
                    break
                if sibling.name:
                    content_list.append(sibling.text.strip())

    if not content_list:
        # Try another article
        return fetch_random_article(base_url, category_path, attempted)

    content = "\n".join(content_list)
    return title, content, image_url, article_url

def generate_tweet_from_lmstudio(title: str, content: str, source_url: str) -> str:
    """Use LM Studio (OpenAI-compatible API) to craft a tweet."""
    if not LM_BASE_URL or not LM_MODEL:
        raise RuntimeError("Missing LMSTUDIO_BASE_URL or LMSTUDIO_MODEL environment variables.")

    client = OpenAI(base_url=LM_BASE_URL, api_key=LM_API_KEY)

    # Leave room for the source URL when posting
    # char_budget = max(120, 280 - len(source_url) - 6)
    char_budget = 3000
    system_msg = """You write social media posts that sound completely natural and human. 
    Your goal is to produce a piece of writing that reads as if it were written by a thoughtful college student, 
    not an AI.

Your writing style:
- Sound like a thoughtful research expert explaining what they learned to the world.
- Use simple, everyday language. Keep sentences short and easy to follow.
- Be direct and get to the point quickly. No fluff.
- Be genuine - don't force excitement or fake enthusiasm.
- Cut unnecessary adjectives and adverbs.
- Be honest about the topic without overselling or hyping it up.
- Avoid sounding like marketing copy.
- Keep a natural flow. Use plain transitions.

Hard constraints:
- No hashtags.
- No emojis.

NEVER use these AI-sounding phrases:
- "Let's dive into", "Unleash", "Game-changing", "Revolutionary"
- "Transform your", "Unlock the secrets", "Leverage", "Optimize"
- "Did you know?", "Here's why", "The truth about"

Instead, write like a real person sharing something interesting they learned.
Your post should sound like something you'd actually say out loud to someone."""

    user_msg = f"""Write a single tweet about this topic:

Title: {title or 'Untitled'}

Key points:
{content}

Rules:
- You don't have a 280 character limit, you just need to stay under {char_budget} characters
- No URL (it gets added separately)
- No hashtags
- No markdown or quotes - rewrite in your own words.
- Include at least 1 specific, concrete detail from the key points (not abstract fluff).
- Just return the tweet text, nothing else.

Don't start the tweet with phrases like "I just read" or "I've been reading about...". Just start with the interesting detail you found.

Write it like a real person who genuinely finds this interesting, not like marketing copy."""

    resp = client.chat.completions.create(
        model=LM_MODEL,
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        temperature=0.7,
        max_tokens=1600,
    )
    tweet = resp.choices[0].message.content.strip()
    # Hard cap to keep room for the URL when combined
    return tweet[:char_budget]


def post_to_x_via_android(final_text: str) -> None:
    """Automate the X Android app to publish the given text."""
    if not ANDROID_SERIAL:
        raise RuntimeError("Set GALAXY_IP to your device serial/IP before running.")

    d = u2.connect(ANDROID_SERIAL)

    # Ensure the app is not already running so we land on the main screen
    d.app_stop("com.twitter.android")
    time.sleep(1)

    d.screen_on()
    d.press("home")

    # Launch X
    d.app_start("com.twitter.android")
    d.app_wait("com.twitter.android", front=True, timeout=10)

    # Tap compose (FAB)
    d(resourceId="com.twitter.android:id/composer_write").click()

    # Wait for compose screen to load - this was the issue!
    time.sleep(3)  # Increased from 1 to 3 seconds for reliability

    d(resourceId="com.twitter.android:id/composer_write").click()

    # Wait for text input elements to appear (up to 5 seconds)
    max_wait = 5
    wait_time = 0
    inputs_found = False
    while wait_time < max_wait:
        if d(classNameMatches=".*EditText|.*TextInput.*|.*MultiAutoCompleteTextView.*").exists:
            inputs_found = True
            break
        time.sleep(0.5)
        wait_time += 0.5

    # Enter text
    if d(resourceId="com.twitter.android:id/tweet_text").exists:
        d(resourceId="com.twitter.android:id/tweet_text").set_text(final_text)
    elif d(className="android.widget.EditText").exists:
        d(className="android.widget.EditText").set_text(final_text)
    elif inputs_found:
        # If we found input elements but not the specific selectors, try the first available input
        all_inputs = d(classNameMatches=".*EditText|.*TextInput.*|.*MultiAutoCompleteTextView.*")
        if len(all_inputs) > 0:
            all_inputs[0].set_text(final_text)
        else:
            raise RuntimeError("Tweet text field not found. Inspect the editor element.")
    else:
        raise RuntimeError("Tweet text field not found. Inspect the editor element.")

    # Tap Post
    d(resourceId="com.twitter.android:id/button_tweet").click()

    # Give the app a moment to send the tweet
    time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description="Scrape articles and post to X")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Output scraped content and generated post without posting to Android",
    )
    args = parser.parse_args()

    print("Scraping article...")
    title, content, image_url, article_url = fetch_random_article(BASE_URL, CATEGORY_PATH)
    if not (title and content and article_url):
        raise RuntimeError("Failed to scrape article content.")

    print(f"Got article: {title} ({article_url})")
    if image_url:
        print(f"Image URL: {image_url}")

    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN MODE - Web Scraped Content")
        print("=" * 60)
        print(f"\nTitle: {title}")
        print(f"\nArticle URL: {article_url}")
        print(f"\nImage URL: {image_url if image_url else 'None'}")
        print(f"\nContent:\n{'-' * 60}")
        print(content)
        print("-" * 60)

    print("\nGenerating tweet with LM Studio...")
    tweet_body = generate_tweet_from_lmstudio(title, content, article_url)
    final_text = f"{tweet_body}\n\n{article_url}".strip()
    if len(final_text) > 3000:
        final_text = final_text[:3000] + "..."

    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN MODE - Model Generated Post")
        print("=" * 60)
        print(f"\nTweet Body (length: {len(tweet_body)}):")
        print(tweet_body)
        print(f"\nFinal Post (length: {len(final_text)}):")
        print(final_text)
        print("=" * 60)
        print("\n[DRY RUN] Skipping Android posting")
        print("Done.")
    else:
        print("\n--- Tweet preview ---")
        print(final_text)
        print("---------------------\n")

        print("Posting to X via Android...")
        post_to_x_via_android(final_text)
        print("Done.")


if __name__ == "__main__":
    main()

