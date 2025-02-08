# PydanticAI running on cron with Github Actions Example

[Accompanying Blog Post](https://christophergs.com/blog/pydantic-ai-example-github-actions)

An example showing the power of PydanticAI as a CLI app. Fetches EV charging station
data and compares against the price of Platinum to seek trading signals. No, the strategy
doesn't work. Yes, it is fun.

## Local Setup

First time:

- Create virtual environment and activate it
- Generate these 2 **free** API keys and set them in .env:
  - [OPENCHARGEMAP_API_KEY](https://openchargemap.org/site/develop/api#/operations/get-openapi)
  - [MARKETSTACK_API_KEY](https://marketstack.com/)
- Add your LLM API key - currently set to `OPENAI_API_KEY` (update settings.py if you're using
a different provider)

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m main --months 3 --metals-etf PPLT
```

Run integration tests (n.b. these make real API calls) via:
```bash
python -m pytest tests/integration/
```

## Why This Is A Fun Pattern

Clearly, this is not a viable investment advice app. It is designed to be illustrative.
This approach holds vast possibility - LLM CLI apps triggered via gh actions cron jobs to 
automate whatever you want. No server to manage means it's particularly
easy. It's free as in beer, apart from the LLM API calls, which if you are sensible
should be less than 5 dollars a month (and if you are careful just a few pennies).

Random cool ideas:
- Automate checking your bank statements for weird/recurring costs
- Automate checking your emails for things you forgot to reply to
- Automate a hacker news scan to highlight articles of things you are interested in


## Obvious Additions
- Setup an email/slack/discord notification
- Monitor with [Pydantic Logfire](https://logfire.pydantic.dev)