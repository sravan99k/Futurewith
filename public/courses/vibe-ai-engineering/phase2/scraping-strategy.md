# Scraping Strategy: Data Ingestion

Phase 2 is about building the bridge between the raw web and your AI logic.

## 1. Industrial Role: Research Engineer
Your job is to gather the "Knowledge Base" for the AI. You are a digital architect building a bridge to the raw web.

## 2. Requirements (Inputs)
- **Target URL List**: High-quality, verified sources of truth.
- **Data Schema**: A clear definition of what fields you need to extract.

## 3. The Industrial Stack
- **Core Tech**: HTTP APIs, Proxies, Python/Node scripts.
- **AI Tools**: **Firecrawl** (for clean Markdown), **Jina Reader**.

## 4. The Industrial Task
- Determine if a site needs "Deep Scraping" (SPAs, JS-heavy) or "Light Reading" (Blogs, Docs).
- Map the data schema that your Logic Engine (Phase 4) will eventually need.

## 5. Why We Scrape
AI can't hallucinate if it has the truth. By scraping raw documentation or data, you give the LLM a "Context Window" of reality.

## Exercise: Identifying Sources
List 5 URLs that contain the "Source of Truth" for your project. Are they protected by Cloudflare? How will you bypass it?
