# AI Job Email Assistant

A local FastAPI web app that parses your resume, scans Gmail inbox mail, scores job-related messages against your profile using an editable agent prompt, and shows the best matches in a modern dashboard.

## Project Structure

```text
email_agent/
├── app.py                 # FastAPI entry point (create_app)
├── api/                   # HTTP routes (thin)
├── services/              # Sync, analysis, draft orchestration
├── llm/                   # Shared OpenAI client + retries
├── constants.py           # Shared statuses, scores, labels
├── errors.py              # Domain exceptions
├── retry.py               # Transient retry helper
├── logging_config.py      # Logging setup
├── gmail_client.py        # Gmail OAuth + API
├── storage.py             # SQLite persistence
├── ai_agent.py            # Email analysis / drafting
├── resume_parser.py       # Resume parsing
├── profile_store.py       # Profile JSON store
├── prompt_store.py        # Agent prompt store
├── static/                # Frontend assets
└── templates/             # HTML shell
```

## Setup

1. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

2. Add your OpenAI settings to `.env`:

   ```env
   OPENAI_API_KEY=your_key_here
   OPENAI_MODEL=gpt-4o-mini
   EMAIL_AGENT_BASE_URL=http://127.0.0.1:8000
   EMAIL_AGENT_LOG_LEVEL=INFO
   ```

3. In Google Cloud Console, make sure the OAuth client allows this redirect URI:

   ```text
   http://127.0.0.1:8000/oauth2callback
   ```

   The app automatically enables OAuthlib's local HTTP exception for `127.0.0.1` or `localhost` development callbacks.

4. Run the app from this folder:

   ```powershell
   python -m uvicorn app:app --reload
   ```

5. Open `http://127.0.0.1:8000`, connect Gmail, and upload your resume. Inbox sync starts automatically.

## Behavior

- Resume upload supports PDF, DOCX, and text files. The app extracts resume text, builds structured profile data, saves it in SQLite under `resume_profiles`, and displays a natural-language profile summary.
- Gmail access uses OAuth and the Gmail API only. The app always syncs `in:inbox newer_than:30d` and stores emails in SQLite.
- On first open, all inbox mail from the last 30 days is fetched. On later opens, only new mail since the last saved email is synced.
- Sync and AI analysis run automatically when you open the app (if Gmail is connected).
- Email statuses are `Not Analyzed`, `Analyzing`, `Analyzed`, `Relevant`, and `Not Relevant`.
- AI matching returns company, job title, job type, match score, confidence, required skills, matched skills, missing skills, summary, and explanation.
- The **Agent Prompt** section lets you customize how mail is classified, scored against your resume, and how replies are drafted. The default focuses on job search. Saved prompts live in `data/agent_prompts.json`.
- Changing the agent prompt queues all saved emails for re-analysis. Click **Sync** to re-run classification and scoring with the new instructions.
- If OpenAI is not configured, the app falls back to deterministic keyword and skill-overlap matching.
- Replies are still never sent automatically. You must review and approve each draft in the UI.
- Runtime data is stored locally under `data/` and ignored by git.
- The latest parsed resume profile is available at `/api/resume-profile` after an upload.
- Gmail and LLM calls retry automatically on transient errors (rate limits, timeouts, 5xx).

## Dashboard Flow

1. Upload your resume and save your profile.
2. Edit the **Agent Prompt** if needed, then click **Save Prompt**.
3. Open the app — it auto-syncs your last 30 days of inbox mail. Use **Sync** anytime for a manual refresh.
4. Sort by recent mail, highest match %, or most relevant.
5. Open an email to see AI confidence vs your profile, then generate a reply if needed.

## Useful Environment Variables

- `OPENAI_API_KEY`: required for standard OpenAI analysis and drafting.
- `OPENAI_MODEL`: optional, defaults to `gpt-4o-mini`.
- `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `DEPLOYMENT_NAME`, `OPENAI_API_VERSION`: supported for Azure OpenAI.
- `EMAIL_AGENT_BASE_URL`: optional, defaults to `http://127.0.0.1:8000`.
- `GMAIL_REDIRECT_URI`: optional override for OAuth callback.
- `EMAIL_AGENT_LOG_LEVEL`: optional, defaults to `INFO`.
