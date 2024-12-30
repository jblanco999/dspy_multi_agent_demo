import requests
from newsapi import NewsApiClient
import dspy
import asyncio
import nest_asyncio
from crawl4ai import AsyncWebCrawler, CacheMode
import time
from typing import Literal
import json

class article_topic_assessment(dspy.Signature):
    """Review the content of the article and send them to a topic specific agent"""

    article_details: dict = dspy.InputField()
    transfer_agent_name: Literal['sports_agent','politics_agent','tech_agent','other_agent'] = dspy.OutputField()

class sports_agent(dspy.Signature):
    """You are a professional sports expert with complete knowledge of all sports. Scrape the full content of the article url using the scrape_tool and determine the bias level from 0 to 100 with 0 being most objective"""

    article_url: str = dspy.InputField()
    scraped_content: str = dspy.OutputField(description="Put the entire scraped article from the scrape_tool here")
    bias: int = dspy.OutputField()
    article_sentiment: Literal['neutral', 'positive', 'negative'] = dspy.OutputField()
    summary_assessment: str = dspy.OutputField()

class politics_agent(dspy.Signature):
    """You are a political genius with the most objective viewpoints possible. Scrape the full content of the article url using the scrape_tool and determine the bias level from 0 to 100 with 0 being most objective"""

    article_url: str = dspy.InputField()
    scraped_content: str = dspy.OutputField(description="Put the entire scraped article from the scrape_tool here")
    bias: int = dspy.OutputField()
    article_sentiment: Literal['neutral', 'positive', 'negative'] = dspy.OutputField()
    summary_assessment: str = dspy.OutputField()

class tech_agent(dspy.Signature):
    """You are a tech genius with knowledge over everything related to technical news. Scrape the full content of the article url using the scrape_tool and determine the bias level from 0 to 100 with 0 being most objective"""

    article_url: str = dspy.InputField()
    scraped_content: str = dspy.OutputField(description="Put the entire scraped article from the scrape_tool here")
    bias: int = dspy.OutputField()
    article_sentiment: Literal['neutral', 'positive', 'negative'] = dspy.OutputField()
    summary_assessment: str = dspy.OutputField()

class other_agent(dspy.Signature):
    """You are a general knowledge agent that generally understands news. Scrape the full content of the article url using the scrape_tool and determine the bias level from 0 to 100 with 0 being most objective. Provide a summary and assessment of the article"""

    article_url: str = dspy.InputField()
    scraped_content: str = dspy.OutputField(description="Put the entire scraped article from the scrape_tool here")
    bias: int = dspy.OutputField()
    article_sentiment: Literal['neutral', 'positive', 'negative'] = dspy.OutputField()
    summary_assessment: str = dspy.OutputField()

class news_analysis(dspy.Signature):
    """Do a detailed assessment the top headlining news for the day and give a summary of general news topics, bias levels and overall sentiment"""

    articles: dict = dspy.InputField()
    articles_assessment: str = dspy.OutputField()

class news_consolidation(dspy.Module):
    def __init__(self):
        self.article_topic_assessment = dspy.Predict(article_topic_assessment)
        self.sports_agent = dspy.ReAct(sports_agent, tools=[self.scrape_tool], max_iters=1)
        self.politics_agent = dspy.ReAct(politics_agent, tools=[self.scrape_tool], max_iters=1)
        self.tech_agent = dspy.ReAct(tech_agent, tools=[self.scrape_tool], max_iters=1)
        self.other_agent = dspy.ReAct(other_agent, tools=[self.scrape_tool], max_iters=1)

    async def async_call_scrape_tool(self, url: str):
        async with AsyncWebCrawler(browser_type="chromium", verbose=True, headless=True) as crawler:
            result = await crawler.arun(url=url, cache_mode=CacheMode.BYPASS, excluded_tags=['form', 'nav', 'header', 'footer'], remove_overlay_elements=True, exclude_external_links=True, exclude_external_images=True)
        return result.markdown

    def scrape_tool(self, url: str):
        nest_asyncio.apply()
        result = asyncio.run(self.async_call_scrape_tool(url))
        return result

    def forward(self, article_content):
        i = 0
        while i < len(article_content):
        # while i < 2:
            time.sleep(30)
            next_agent = self.article_topic_assessment(article_details=article_content[i])
            print(next_agent)
            time.sleep(30)
            if next_agent.transfer_agent_name == 'other_agent':
                result = self.other_agent(article_url=article_content[i]['url'])
                article_content[i]['topic'] = 'other'
                article_content[i]['content'] = result.trajectory['observation_0']
                article_content[i]['bias'] = result.bias
                article_content[i]['sentiment'] = result.article_sentiment
                article_content[i]['summary_and_assessment'] = result.summary_assessment
                print(result)
                i+=1
            if next_agent.transfer_agent_name == 'sports_agent':
                result = self.sports_agent(article_url=article_content[i]['url'])
                article_content[i]['topic'] = 'sports'
                article_content[i]['content'] = result.trajectory['observation_0']
                article_content[i]['bias'] = result.bias
                article_content[i]['sentiment'] = result.article_sentiment
                article_content[i]['summary_and_assessment'] = result.summary_assessment

                print(result)
                i+=1
            if next_agent.transfer_agent_name == 'tech_agent':
                result = self.tech_agent(article_url=article_content[i]['url'])
                article_content[i]['topic'] = 'technology'
                article_content[i]['content'] = result.trajectory['observation_0']
                article_content[i]['bias'] = result.bias
                article_content[i]['sentiment'] = result.article_sentiment
                article_content[i]['summary_and_assessment'] = result.summary_assessment
                print(result)
                i+=1
            if next_agent.transfer_agent_name == 'politics_agent':
                result = self.politics_agent(article_url=article_content[i]['url'])
                article_content[i]['topic'] = 'politics'
                article_content[i]['content'] = result.trajectory['observation_0']
                article_content[i]['bias'] = result.bias
                article_content[i]['sentiment'] = result.article_sentiment
                article_content[i]['summary_and_assessment'] = result.summary_assessment
                print(result)
                i+=1
        return article_content

def get_news():
    newsapi = NewsApiClient(api_key='7b0a505552c34422a83e27dd7bfec465')

    url = "https://newsapi.org/v2/everything"
    api_key = "7b0a505552c34422a83e27dd7bfec465"

    params = {
        "q": "all",  # Your search query
        "from": "2024-12-29",  # Start date
        "to": "2024-12-29",  # End date
        "sortBy": "publishedAt",  # Sort by publication date
        "language": "en",  # Article language
        "sources": 'bbc-news,the-verge,abc-news,cnn,yahoo,vox,cbs-news,fox-news',
        "apiKey": api_key
    }

    response = requests.get(url, params=params)
    top_headlines = response.json().get("articles", [])
    top_headlines = top_headlines[0:50]

    return top_headlines

def set_dspy_llm():
    llama = dspy.LM('databricks/databricks-meta-llama-3-1-70b-instruct', cache=False)
    mixtral = dspy.LM('databricks/databricks-mixtral-8x7b-instruct')
    gpt4o = dspy.LM('openai/gpt-4o', cache=False)
    dbrx = dspy.LM('databricks/databricks-dbrx-instruct')
    claude = dspy.LM('anthropic/claude-3-5-sonnet-20241022', max_tokens=8000, cache=False)
    dspy.configure(lm=claude)

def main():
    i = 0
    article_content = []
    top_headlines = get_news()
    while i < len(top_headlines):
        title = top_headlines[i]['title']
        url = top_headlines[i]['url']
        content = top_headlines[i]['content']
        date = top_headlines[i]['publishedAt']
        article_content.append({'title': title, 'url': url, 'content': content, 'date': date})
        i += 1
    print(f"Total Articles: {len(article_content)}\n\n")
    set_dspy_llm()
    consolidation_test = news_consolidation()
    print("Starting LLM runs: \n\n")
    start = time.time()
    result = consolidation_test(article_content=article_content)
    end = time.time()
    elapsed = end - start
    print(f"Total Time: {elapsed:.2f} seconds")
    with open("output.json", "w") as f:
        json.dump(result, f)
    the_analysis = dspy.Predict(news_analysis)
    the_analysis_output = the_analysis(articles=result)
    with open("output_assessment.txt", "w") as f:
        f.write(the_analysis_output.articles_assessment)

if __name__ == "__main__":
    main()