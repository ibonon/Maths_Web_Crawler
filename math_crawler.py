import asyncio
import re
import math
import sqlite3
import json
from collections import defaultdict, deque
from urllib.parse import urljoin, urlparse

# crawl4ai imports - THIS IS THE CORE CRAWLING ENGINE
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

# --- 1. Database & Search Engine (Optimisé BM25 avec Pondération) ---
class MathSearchIndex:
    def __init__(self, db_path='math_search.db', k1=1.2, b=0.75):
        self.conn = sqlite3.connect(db_path)
        self._init_db()
        
        # BM25 Parameters
        self.k1 = k1
        self.b = b
        self.avgdl = 0.0 
        self.total_docs = 0
        self.total_doc_length = 0
        
        # Index structures
        self.index = defaultdict(list)
        self.docs = {}
        self.doc_freq = defaultdict(int)
        
        # Field weights for scoring (Title > Formulas > Content)
        self.field_weights = {'title': 3.0, 'formulas': 2.0, 'content': 1.0}

    def _init_db(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            url TEXT,
            title TEXT,
            content TEXT,
            formulas TEXT,
            category TEXT,
            links TEXT
        )''')
        self.conn.commit()

    def load_existing_index(self):
        c = self.conn.cursor()
        print("Loading index from database...")
        rows = c.execute('SELECT id, url, title, content, formulas FROM documents').fetchall()
        
        for row in rows:
            doc_id, url, title, content, formulas = row
            data = {'url': url, 'title': title, 'content': content, 'formulas': formulas}
            self._add_to_memory_index(doc_id, data)
        
        if self.total_docs > 0:
            self.avgdl = self.total_doc_length / self.total_docs
            print(f"Index loaded. {self.total_docs} documents (Avg. Length: {self.avgdl:.2f}) ready.")
        else:
            print("Index loaded. No documents found.")

    def _tokenize(self, text):
        return re.findall(r'\w+', text.lower())

    def _calculate_field_tf_and_text(self, text):
        words = self._tokenize(text)
        tf = defaultdict(int)
        for word in words:
            tf[word] += 1
        return tf, len(words), text

    def _add_to_memory_index(self, doc_id, data):
        if doc_id in self.docs:
            return

        doc_data = {'url': data['url'], 'fields': {}}
        total_doc_length = 0
        
        for field, text in [(f, data.get(f, '')) for f in self.field_weights.keys()]:
            tf, length, original_text = self._calculate_field_tf_and_text(text)
            doc_data['fields'][field] = {'tf': tf, 'length': length, 'text': original_text}
            total_doc_length += length

        doc_data['total_length'] = total_doc_length
        self.docs[doc_id] = doc_data
        self.total_doc_length += total_doc_length
        self.total_docs += 1
        
        unique_words = set(w for field_data in doc_data['fields'].values() for w in field_data['tf'].keys())
        for word in unique_words:
            self.index[word].append(doc_id)
            self.doc_freq[word] += 1
            
    def add_document(self, doc_id, data):
        self._add_to_memory_index(doc_id, data)
        
        if self.total_docs > 0:
            self.avgdl = self.total_doc_length / self.total_docs

        c = self.conn.cursor()
        c.execute('''INSERT OR REPLACE INTO documents 
                     (id, url, title, content, formulas, category, links) 
                     VALUES (?, ?, ?, ?, ?, ?, ?)''', (
            doc_id, 
            data['url'], 
            data.get('title', ''), 
            data.get('content', ''), 
            data.get('formulas', ''),
            data.get('category', 'General'),
            json.dumps(data.get('links', []))
        ))
        self.conn.commit()


    def search(self, query, top_k=5):
        words = self._tokenize(query)
        if not words or self.total_docs == 0: 
            return []

        if self.avgdl == 0 and self.total_docs > 0:
            self.avgdl = self.total_doc_length / self.total_docs

        scores = defaultdict(float)
        
        for word in words:
            if word not in self.index: continue
            
            # BM25 IDF
            N = self.total_docs
            df = self.doc_freq[word]
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            
            relevant_docs = set(self.index[word])
            
            for doc_id in relevant_docs:
                doc_data = self.docs[doc_id]
                score_doc = 0.0
                
                # Field Weighting and BM25 Scoring
                for field, weight in self.field_weights.items():
                    field_data = doc_data['fields'][field]
                    tf = field_data['tf'].get(word, 0)
                    
                    if tf == 0: continue

                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_data['total_length'] / self.avgdl))
                    
                    score_doc += weight * idf * (numerator / denominator)
                
                scores[doc_id] += score_doc

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, score in ranked:
            doc = self.docs[doc_id]
            original_text = " ".join([doc['fields'][f]['text'] for f in self.field_weights.keys()])
            
            results.append({
                'url': doc['url'], 
                'score': score, 
                'snippet': self._generate_snippet(original_text, words)
            })
            
        return results

    def _generate_snippet(self, text, query_words, window=150):
        lower_text = text.lower()
        start_idx = -1
        for qw in query_words:
            start_idx = lower_text.find(qw)
            if start_idx != -1: break
            
        if start_idx == -1: return text[:window] + "..."
        
        start = max(0, start_idx - 50)
        end = min(len(text), start_idx + window)
        return "..." + text[start:end] + "..."


# --- 2. The Powerful Crawler (BFS & Discovery) ---
class MathCrawler:
    def __init__(self, index, max_pages=15):
        self.index = index
        self.visited = set()
        self.queue = deque()
        self.max_pages = max_pages
        self.crawled_count = 0
        
        # Schema uses CSS selectors to extract structured math content
        self.schema = {
            "name": "Math Content & Links",
            "baseSelector": "body",
            "fields": [
                {"name": "title", "selector": "h1, title", "type": "text"},
                {"name": "content", "selector": "p, li, dd", "type": "text"},
                # Target common math rendering classes (LaTeX, MathML)
                {"name": "formulas", "selector": ".mwe-math-element, math, .tex, .katex-mathml", "type": "text"},
                # Link discovery for recursion
                {"name": "links", "selector": "a", "type": "attribute", "attribute": "href"}
            ]
        }

    def is_valid_math_url(self, url):
        """Filter to ensure we stay on math topics and known domains (e.g., Wikipedia)."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        if 'wikipedia.org' not in parsed.netloc:
            return False

        if path.endswith(('.jpg', '.png', '.pdf', '.css', '.js', '.svg', '#')): return False
        
        keywords = ['math', 'wiki', 'definition', 'théorème', 'proof', 'algebra', 'calculus', 'geometry', 'analyse', 'espace', 'fonction']
        return any(k in url.lower() for k in keywords)

    async def run(self, seed_urls):
        for url in seed_urls:
            self.queue.append(url)

        browser_config = BrowserConfig(headless=True, verbose=False)
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.ENABLED,
            word_count_threshold=50
        )

        crawler = AsyncWebCrawler(config=browser_config)
        await crawler.start()

        print(f"🚀 Starting crawl (Limit: {self.max_pages} pages)...")

        try:
            while self.queue and self.crawled_count < self.max_pages:
                url = self.queue.popleft()
                
                if url in self.visited:
                    continue
                
                self.visited.add(url)
                self.crawled_count += 1
                
                print(f"[{self.crawled_count}/{self.max_pages}] Crawling: {url}")

                # Core action: Fetching with crawl4ai
                result = await crawler.arun(
                    url, 
                    config=run_config,
                    extraction_strategy=JsonCssExtractionStrategy(schema=self.schema)
                )

                if not result.success:
                    print(f"Failed to crawl {url}")
                    continue

                # Parse and process
                try:
                    data = json.loads(result.extracted_content)
                except json.JSONDecodeError:
                    print(f"Failed to decode JSON from {url}")
                    continue
                
                clean_data = {
                    "url": url, "title": "", "content": "", "formulas": "", "links": []
                }
                
                for item in data:
                    if item.get('title'): clean_data['title'] = item['title'].strip().replace('\n', ' ')
                    if item.get('content'): clean_data['content'] += " " + item['content'].strip().replace('\n', ' ')
                    if item.get('formulas'): clean_data['formulas'] += " " + item['formulas'].strip().replace('\n', ' ')
                    if item.get('links'): 
                        if isinstance(item['links'], list):
                            clean_data['links'].extend(item['links'])
                        elif isinstance(item['links'], str):
                            clean_data['links'].append(item['links'])

                # Indexation (using BM25)
                self.index.add_document(doc_id=url, data=clean_data)

                # Discovery (BFS)
                for link in clean_data['links']:
                    absolute_link = urljoin(url, link)
                    if self.is_valid_math_url(absolute_link) and absolute_link not in self.visited:
                        self.queue.append(absolute_link)

                await asyncio.sleep(0.5)

        finally:
            await crawler.close()
            print("🛑 Crawling finished.")

# --- 3. Main Orchestrator ---
async def main():
    index = MathSearchIndex()
    
    print("="*40)
    print("    POWER MATH CRAWLER (BM25 Indexer)   ")
    print("="*40)
    print("1. Crawl and Index (Build DB)")
    print("2. Search Existing Index (Use DB)")
    choice = input("Select option (1/2): ")

    if choice == '1':
        seeds = [
            "https://fr.wikipedia.org/wiki/Math%C3%A9matiques",
            "https://fr.wikipedia.org/wiki/Th%C3%A9or%C3%A8me",
            "https://fr.wikipedia.org/wiki/Alg%C3%A8bre"
        ]
        crawler = MathCrawler(index, max_pages=15) 
        await crawler.run(seeds)
        index.load_existing_index() 
    else:
        index.load_existing_index()

    while True:
        print("\n" + "="*30)
        query = input("🔍 Enter math query (or 'q' to quit): ")
        if query.lower() in ['q', 'exit', 'quit']:
            break
            
        results = index.search(query)
        
        if not results:
            print("No results found.")
        else:
            print(f"Found {len(results)} results (BM25 Score):\n")
            for i, res in enumerate(results, 1):
                print(f"{i}. Score: {res['score']:.4f}")
                print(f"   URL: {res['url']}")
                print(f"   Excerpt: {res['snippet']}")
                print("-" * 20)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting.")
