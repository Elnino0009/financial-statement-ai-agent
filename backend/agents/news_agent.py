import os
import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import re
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """Data structure for a news article"""
    title: str
    description: str
    url: str
    published_at: str
    source: str
    relevance_score: float = 0.0
    sentiment: str = "neutral"

class NewsAgent:
    """
    News Agent that fetches and analyzes financial news for assets using free APIs
    """
    
    def __init__(self):
        # Initialize OpenAI client with specific model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = "text-embedding-3-small"  # Cost-effective, good performance
        self.chat_model = "gpt-4.1-mini-2025-04-14"      # GPT-4.1 mini for enhanced quality
        
        # News API configurations (free tiers)
        self.news_apis = {
            'newsapi': {
                'key': os.getenv("NEWS_API_KEY"),
                'base_url': 'https://newsapi.org/v2',
                'daily_limit': 500  # Free tier limit
            },
            'alpha_vantage': {
                'key': os.getenv("ALPHA_VANTAGE_KEY"),
                'base_url': 'https://www.alphavantage.co/query',
                'daily_limit': 100
            }
        }
        
        # Cache for rate limiting and efficiency
        self.news_cache = {}  # asset -> cached news
        self.rate_limits = {}  # api -> usage tracking
        
        # System prompt optimized for financial news analysis
        self.system_prompt = """You are a specialized Financial News Analyst AI. Your expertise includes:

**Core Capabilities:**
1. **Market Impact Analysis**: Assess how news affects asset prices and market sentiment
2. **Risk Assessment**: Identify potential risks and opportunities from news events
3. **Sentiment Analysis**: Analyze market sentiment from news tone and content
4. **Relevance Scoring**: Determine news relevance to specific assets and portfolios

**Analysis Framework:**
- **Immediate Impact**: Short-term price/volatility implications
- **Long-term Implications**: Strategic and fundamental impacts
- **Risk Factors**: Regulatory, competitive, or market risks
- **Opportunities**: Growth, expansion, or strategic advantages

**Response Format:**
- Provide clear, actionable insights
- Include sentiment assessment (Positive/Negative/Neutral)
- Highlight key risk factors and opportunities
- Use specific examples from the news content
- Keep analysis concise but comprehensive"""

    async def get_asset_news(self, assets: List[str], session_id: str, days_back: int = 7) -> Dict:
        """
        Main method to fetch and analyze news for given assets
        
        Why this approach:
        1. Parallel processing for multiple assets improves speed
        2. Rate limiting prevents API quota exhaustion
        3. Caching reduces redundant API calls
        4. Multiple news sources provide comprehensive coverage
        """
        try:
            logger.info(f"Fetching news for assets: {assets}")
            
            # Step 1: Validate inputs and check cache
            if not assets:
                return {"status": "error", "message": "No assets provided"}
            
            # Step 2: Fetch news from multiple sources in parallel
            news_tasks = []
            for asset in assets:
                # Create async tasks for each asset to improve performance
                news_tasks.append(self._fetch_asset_news(asset, days_back))
            
            # Execute all news fetching tasks concurrently
            news_results = await asyncio.gather(*news_tasks, return_exceptions=True)
            
            # Step 3: Aggregate and filter news
            all_articles = []
            asset_news_map = {}
            
            for i, result in enumerate(news_results):
                asset = assets[i]
                if isinstance(result, Exception):
                    logger.error(f"Error fetching news for {asset}: {result}")
                    asset_news_map[asset] = {"error": str(result)}
                else:
                    asset_news_map[asset] = result
                    all_articles.extend(result.get("articles", []))
            
            # Step 4: Generate comprehensive analysis using GPT-4.1 nano
            analysis = await self._generate_portfolio_news_analysis(assets, asset_news_map)
            
            return {
                "status": "success",
                "assets_analyzed": assets,
                "total_articles": len(all_articles),
                "asset_news": asset_news_map,
                "portfolio_analysis": analysis,
                "generated_at": datetime.now().isoformat(),
                "model_used": self.chat_model
            }
            
        except Exception as e:
            logger.error(f"Error in get_asset_news: {e}")
            return {"status": "error", "message": str(e)}

    async def _fetch_asset_news(self, asset: str, days_back: int) -> Dict:
        """
        Fetch news for a specific asset from multiple sources
        
        Why multiple sources:
        1. Different APIs have different coverage strengths
        2. Redundancy ensures we get news even if one API fails
        3. Cross-validation improves accuracy
        4. Free tier limits require smart distribution
        """
        # Check cache first to avoid unnecessary API calls
        cache_key = f"{asset}_{days_back}"
        if cache_key in self.news_cache:
            cached_time = self.news_cache[cache_key].get("fetched_at")
            if cached_time and (datetime.now() - datetime.fromisoformat(cached_time)).hours < 1:
                logger.info(f"Using cached news for {asset}")
                return self.news_cache[cache_key]
        
        articles = []
        
        # Source 1: NewsAPI.org (best for general news)
        if self._can_use_api('newsapi'):
            newsapi_articles = await self._fetch_from_newsapi(asset, days_back)
            articles.extend(newsapi_articles)
        
        # Source 2: Alpha Vantage (good for financial-specific news)
        if self._can_use_api('alpha_vantage'):
            av_articles = await self._fetch_from_alpha_vantage(asset, days_back)
            articles.extend(av_articles)
        
        # Remove duplicates and sort by relevance
        unique_articles = self._deduplicate_articles(articles)
        
        # Score articles for relevance using embeddings
        scored_articles = await self._score_article_relevance(unique_articles, asset)
        
        # Take top 10 most relevant articles
        top_articles = sorted(scored_articles, key=lambda x: x.relevance_score, reverse=True)[:10]
        
        result = {
            "asset": asset,
            "articles": [self._article_to_dict(article) for article in top_articles],
            "total_found": len(unique_articles),
            "fetched_at": datetime.now().isoformat()
        }
        
        # Cache the result
        self.news_cache[cache_key] = result
        
        return result

    async def _fetch_from_newsapi(self, asset: str, days_back: int) -> List[NewsArticle]:
        """
        Fetch news from NewsAPI.org
        
        Why NewsAPI.org:
        1. 500 free requests per day
        2. Good coverage of financial news
        3. Reliable API with good documentation
        4. Supports advanced query parameters
        """
        articles = []
        
        try:
            api_key = self.news_apis['newsapi']['key']
            if not api_key:
                logger.warning("NewsAPI key not configured")
                return articles
            
            # Calculate date range
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"📅 NewsAPI date range for {asset}: {from_date} to {to_date}")
            
            # Create search query - include company name variations
            query = self._create_search_query(asset)
            
            url = f"{self.news_apis['newsapi']['base_url']}/everything"
            params = {
                'q': query,
                'from': from_date,
                'to': to_date,  # Add explicit end date
                'sortBy': 'publishedAt',  # Sort by most recent first
                'language': 'en',
                'pageSize': 20,  # Limit to control API usage
                'apiKey': api_key
            }
            
            logger.info(f"🌐 NewsAPI request for {asset}: {url} with params: {params}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        logger.info(f"✅ NewsAPI response for {asset}: {data.get('totalResults', 0)} total results")
                        
                        article_list = data.get('articles', [])
                        logger.info(f"📰 NewsAPI articles received for {asset}: {len(article_list)}")
                        
                        # Log sample dates to debug the October 2023 issue
                        if article_list:
                            sample_dates = [art.get('publishedAt', 'No date') for art in article_list[:3]]
                            logger.info(f"🔍 Sample NewsAPI dates for {asset}: {sample_dates}")
                        
                        for article_data in article_list:
                            article = NewsArticle(
                                title=article_data.get('title', ''),
                                description=article_data.get('description', ''),
                                url=article_data.get('url', ''),
                                published_at=article_data.get('publishedAt', ''),
                                source=article_data.get('source', {}).get('name', 'NewsAPI')
                            )
                            articles.append(article)
                        
                        # Track API usage
                        self._track_api_usage('newsapi', 1)
                        logger.info(f"✅ Fetched {len(articles)} articles from NewsAPI for {asset}")
                    
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ NewsAPI error for {asset}: {response.status} - {error_text}")
        
        except Exception as e:
            logger.error(f"❌ Error fetching from NewsAPI for {asset}: {e}")
        
        return articles

    async def _fetch_from_alpha_vantage(self, asset: str, days_back: int) -> List[NewsArticle]:
        """
        Fetch news from Alpha Vantage
        
        Why Alpha Vantage:
        1. Specialized in financial data
        2. Good quality financial news
        3. Free tier available
        4. Integrates well with stock symbols
        """
        articles = []
        
        try:
            api_key = self.news_apis['alpha_vantage']['key']
            if not api_key:
                logger.warning("Alpha Vantage key not configured")
                return articles
            
            # Alpha Vantage uses stock symbols, so we need to extract/guess the symbol
            symbol = self._extract_stock_symbol(asset)
            logger.info(f"📈 Alpha Vantage symbol for {asset}: {symbol}")
            
            url = self.news_apis['alpha_vantage']['base_url']
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': api_key,
                'limit': 20
            }
            
            logger.info(f"🌐 Alpha Vantage request for {asset}: {url} with params: {params}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Check for API errors
                        if "Error Message" in data:
                            logger.error(f"❌ Alpha Vantage API Error for {asset}: {data['Error Message']}")
                            return articles
                        
                        if "Note" in data:
                            logger.warning(f"⚠️ Alpha Vantage Note for {asset}: {data['Note']}")
                            return articles
                        
                        feed_data = data.get('feed', [])
                        logger.info(f"📰 Alpha Vantage articles received for {asset}: {len(feed_data)}")
                        
                        # Log sample dates to debug the October 2023 issue  
                        if feed_data:
                            sample_dates = [art.get('time_published', 'No date') for art in feed_data[:3]]
                            logger.info(f"🔍 Sample Alpha Vantage dates for {asset}: {sample_dates}")
                        
                        for article_data in feed_data:
                            article = NewsArticle(
                                title=article_data.get('title', ''),
                                description=article_data.get('summary', ''),
                                url=article_data.get('url', ''),
                                published_at=article_data.get('time_published', ''),
                                source='Alpha Vantage',
                                sentiment=article_data.get('overall_sentiment_label', 'neutral')
                            )
                            articles.append(article)
                        
                        self._track_api_usage('alpha_vantage', 1)
                        logger.info(f"✅ Fetched {len(articles)} articles from Alpha Vantage for {asset}")
                    
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Alpha Vantage error for {asset}: {response.status} - {error_text}")
        
        except Exception as e:
            logger.error(f"❌ Error fetching from Alpha Vantage for {asset}: {e}")
        
        return articles

    def _create_search_query(self, asset: str) -> str:
        """
        Create optimized search query for news APIs
        
        Why this approach:
        1. Include variations of company names
        2. Add financial keywords for better targeting
        3. Exclude noise (sports, entertainment unless relevant)
        """
        # Basic query with the asset name
        query_parts = [f'"{asset}"']
        
        # Add common financial keywords to improve relevance
        financial_keywords = ['stock', 'shares', 'earnings', 'financial', 'investment', 'market']
        query_parts.extend(financial_keywords)
        
        # Join with OR operators for broader coverage
        return ' OR '.join(query_parts)

    def _extract_stock_symbol(self, asset: str) -> str:
        """
        Extract or guess stock symbol from asset name
        
        Why needed:
        1. Alpha Vantage works better with stock symbols
        2. Many assets might be provided as company names
        3. Improves news relevance
        """
        # Simple symbol extraction (you'd want a more robust mapping in production)
        symbol_patterns = {
            'apple': 'AAPL',
            'microsoft': 'MSFT',
            'google': 'GOOGL',
            'amazon': 'AMZN',
            'tesla': 'TSLA',
            'meta': 'META',
            'nvidia': 'NVDA'
        }
        
        asset_lower = asset.lower()
        for name, symbol in symbol_patterns.items():
            if name in asset_lower:
                return symbol
        
        # If no match, return the asset as-is (might already be a symbol)
        return asset.upper()

    async def _score_article_relevance(self, articles: List[NewsArticle], asset: str) -> List[NewsArticle]:
        """
        Score articles for relevance using text-embedding-3-small
        
        Why use embeddings:
        1. Semantic similarity is more accurate than keyword matching
        2. text-embedding-3-small is cost-effective and fast
        3. Helps filter noise and improve content quality
        4. Enables intelligent ranking of articles
        """
        if not articles:
            return articles
        
        try:
            # Create reference embedding for the asset
            asset_query = f"Financial news about {asset} stock price earnings investment"
            
            # Get embeddings for the asset query and all article texts
            texts_to_embed = [asset_query]  # First item is our reference
            
            for article in articles:
                # Combine title and description for better context
                article_text = f"{article.title} {article.description}"
                texts_to_embed.append(article_text)
            
            # Generate embeddings using text-embedding-3-small
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts_to_embed,
                encoding_format="float"
            )
            
            embeddings = [data.embedding for data in response.data]
            asset_embedding = embeddings[0]  # Reference embedding
            article_embeddings = embeddings[1:]  # Article embeddings
            
            # Calculate similarity scores
            import numpy as np
            
            asset_vector = np.array(asset_embedding)
            
            for i, article in enumerate(articles):
                article_vector = np.array(article_embeddings[i])
                
                # Calculate cosine similarity
                similarity = np.dot(asset_vector, article_vector) / (
                    np.linalg.norm(asset_vector) * np.linalg.norm(article_vector)
                )
                
                # Convert to 0-100 scale
                article.relevance_score = max(0, similarity * 100)
            
            logger.info(f"Scored {len(articles)} articles for relevance to {asset}")
            
        except Exception as e:
            logger.error(f"Error scoring article relevance: {e}")
            # Assign default scores if embedding fails
            for article in articles:
                article.relevance_score = 50.0
        
        return articles

    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """
        Remove duplicate articles based on title similarity
        
        Why deduplication:
        1. Multiple APIs may return the same articles
        2. Reduces noise and improves analysis quality
        3. Saves processing time for LLM analysis
        """
        if not articles:
            return articles
        
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            # Normalize title for comparison
            normalized_title = article.title.lower().strip()
            
            # Simple deduplication - could be improved with fuzzy matching
            if normalized_title not in seen_titles and len(normalized_title) > 10:
                seen_titles.add(normalized_title)
                unique_articles.append(article)
        
        logger.info(f"Deduplicated {len(articles)} -> {len(unique_articles)} articles")
        return unique_articles

    async def _generate_portfolio_news_analysis(self, assets: List[str], asset_news_map: Dict) -> str:
        """
        Generate comprehensive portfolio-level news analysis using GPT-4.1 nano
        
        Why GPT-4.1 nano:
        1. Optimized for speed and cost-efficiency
        2. Good performance for financial analysis tasks
        3. Lower latency for real-time responses
        4. Sufficient context window for multiple articles
        """
        try:
            # Prepare news summary for analysis
            news_summary = []
            
            for asset, news_data in asset_news_map.items():
                if "error" in news_data:
                    continue
                
                articles = news_data.get("articles", [])
                if articles:
                    # Summarize top articles for this asset
                    top_articles = articles[:3]  # Top 3 most relevant
                    
                    asset_summary = f"\n**{asset} News:**\n"
                    for i, article in enumerate(top_articles, 1):
                        asset_summary += f"{i}. {article['title']}\n   {article['description'][:200]}...\n   Relevance: {article['relevance_score']:.1f}%\n\n"
                    
                    news_summary.append(asset_summary)
            
            if not news_summary:
                return "No relevant news found for the specified assets."
            
            # Create comprehensive analysis prompt
            analysis_prompt = f"""
**Portfolio Assets:** {', '.join(assets)}

**Recent News Summary:**
{''.join(news_summary)}

Please provide a comprehensive analysis including:
1. **Overall Market Sentiment** for the portfolio
2. **Key Risks** identified from the news
3. **Opportunities** highlighted in the coverage
4. **Individual Asset Insights** with specific recommendations
5. **Portfolio Impact Assessment** - how this news might affect overall portfolio performance

Focus on actionable insights for investment decision-making."""

            # Generate analysis using GPT-4.1 nano
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1,  # Low temperature for factual analysis
                max_tokens=1000,
                top_p=0.9
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating portfolio analysis: {e}")
            return "Unable to generate portfolio analysis due to a technical error."

    def _can_use_api(self, api_name: str) -> bool:
        """
        Check if we can use an API based on rate limits
        
        Why rate limiting:
        1. Prevents exceeding free tier quotas
        2. Ensures sustainable usage throughout the day
        3. Avoids API blocking
        """
        if api_name not in self.rate_limits:
            self.rate_limits[api_name] = {"count": 0, "date": datetime.now().date()}
        
        rate_limit = self.rate_limits[api_name]
        
        # Reset counter if it's a new day
        if rate_limit["date"] != datetime.now().date():
            rate_limit["count"] = 0
            rate_limit["date"] = datetime.now().date()
        
        # Check if we're under the daily limit
        daily_limit = self.news_apis[api_name]["daily_limit"]
        return rate_limit["count"] < daily_limit

    def _track_api_usage(self, api_name: str, count: int = 1):
        """Track API usage for rate limiting"""
        if api_name not in self.rate_limits:
            self.rate_limits[api_name] = {"count": 0, "date": datetime.now().date()}
        
        self.rate_limits[api_name]["count"] += count

    def _article_to_dict(self, article: NewsArticle) -> Dict:
        """Convert NewsArticle to dictionary for JSON serialization"""
        return {
            "title": article.title,
            "description": article.description,
            "url": article.url,
            "published_at": article.published_at,
            "source": article.source,
            "relevance_score": article.relevance_score,
            "sentiment": article.sentiment
        }

    def get_usage_stats(self) -> Dict:
        """Get current API usage statistics"""
        return {
            "rate_limits": self.rate_limits,
            "cache_size": len(self.news_cache),
            "models_used": {
                "embedding": self.embedding_model,
                "chat": self.chat_model
            }
        }

# Test function
async def test_news_agent():
    """Test the news agent with sample assets"""
    news_agent = NewsAgent()
    
    # Test with sample assets
    test_assets = ["Apple Inc", "Microsoft Corp"]
    
    result = await news_agent.get_asset_news(test_assets, "test_session", days_back=3)
    
    print("News Agent Test Result:")
    print(json.dumps(result, indent=2, default=str))
    
    # Test usage stats
    stats = news_agent.get_usage_stats()
    print("\nUsage Stats:")
    print(json.dumps(stats, indent=2, default=str))

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_news_agent())
